# app.py

import os
import json
import re
import uuid
import logging
from typing import List, Optional, Dict
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from supabase import create_client

# -------------------------
# Logging Configuration
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# Configuration
# -------------------------
class Config:
    def __init__(self):
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.ASSISTANT_ID    = os.getenv("ASSISTANT_ID")  # e.g. "asst_xxx"
        self.SUPABASE_URL    = os.getenv("SUPABASE_URL")
        self.SUPABASE_KEY    = os.getenv("SUPABASE_KEY")
        self.MAX_RETRIES     = int(os.getenv("MAX_RETRIES", "3"))

    def validate(self):
        missing = []
        for var in ["OPENAI_API_KEY", "ASSISTANT_ID", "SUPABASE_URL", "SUPABASE_KEY"]:
            if not getattr(self, var):
                missing.append(var)
        if missing:
            raise ValueError(f"Missing required env vars: {missing}")

config = Config()
try:
    config.validate()
    logger.info("Configuration validated successfully")
except ValueError as e:
    logger.error(f"Configuration error: {e}")
    raise

# -------------------------
# Clients Initialization
# -------------------------
# OpenAI client manager
class ClientManager:
    async def get_openai_client(self) -> AsyncOpenAI:
        try:
            client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
            await client.models.list()  # sanity check
            return client
        except Exception as e:
            logger.error(f"Failed to create OpenAI client: {e}")
            raise HTTPException(status_code=503, detail=f"OpenAI connection failed: {e}")

client_manager = ClientManager()

# Supabase client
supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)

# -------------------------
# Pydantic Models
# -------------------------
class UserProfile(BaseModel):
    dietary_preferences: List[str] = []
    allergies:             List[str] = []
    intolerances:          List[str] = []
    other_flags:           List[str] = []

class RecipeContext(BaseModel):
    title:             str
    description:       str
    ingredients:       List[str]
    preparation:       List[str]
    nutrition_content: Optional[str] = None

class ChefRequest(BaseModel):
    user_id:     str
    recipe_id:   Optional[str]       = None
    usermessage: str
    thread_id:   Optional[str]       = None
    context:     Optional[RecipeContext] = None
    profile:     UserProfile

class RecipeInfo(BaseModel):
    name:         str
    ingredients:  List[str]
    instructions: List[str]
    tips:         Optional[List[str]] = []

class MealPlanDay(BaseModel):
    breakfast: List[RecipeInfo]
    lunch:     List[RecipeInfo]
    dinner:    List[RecipeInfo]
    snacks:    Optional[List[RecipeInfo]] = []

class MealPlan(BaseModel):
    monday:    MealPlanDay
    tuesday:   MealPlanDay
    wednesday: MealPlanDay
    thursday:  MealPlanDay
    friday:    MealPlanDay
    saturday:  MealPlanDay
    sunday:    MealPlanDay

class ChefResponse(BaseModel):
    text:                str
    meal_plan:           Optional[MealPlan]   = None
    recipe:              Optional[RecipeInfo] = None
    recipe_id:           Optional[str]        = None
    follow_up_questions: Optional[List[str]]  = []
    thread_id:           str

# -------------------------
# Helper Functions
# -------------------------
def insert_chat(user_id: str, system_role: str, content: str, recipe_id: Optional[str] = None):
    data = {
        "user_id":  user_id,
        "system":   system_role,
        "content":  content,
        "recipe_id": recipe_id
    }
    supabase.table("chat").insert(data).execute()

async def get_chef_thread_context(thread_id: str, client) -> Dict:
    try:
        msgs = await client.beta.threads.messages.list(thread_id=thread_id, order="asc", limit=100)
        user_msgs = [m.content[0].text.value for m in msgs.data if m.role=="user" and m.content]
        return {"user_messages": user_msgs}
    except:
        return {"user_messages": []}

async def classify_chef_intent(message: str, client) -> str:
    prompt = (
        "You are a classifier for a cooking assistant. "
        "Label the user message with exactly one of: "
        "GREETING, THANKS, RECIPE_REQUEST, MEAL_PLAN_REQUEST, EDIT_RECIPE, SPECIAL_OCCASION_REQUEST, OTHER.\n\n"
        f"User: \"{message.strip()}\""
    )
    resp = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        assistant=config.ASSISTANT_ID,
        messages=[{"role":"user","content":prompt}],
        temperature=0.0, max_tokens=4
    )
    label = resp.choices[0].message.content.strip().upper()
    valid = {"GREETING","THANKS","RECIPE_REQUEST","MEAL_PLAN_REQUEST","EDIT_RECIPE","SPECIAL_OCCASION_REQUEST","OTHER"}
    return label if label in valid else "OTHER"

async def generate_chef_follow_up_questions(context: Dict, client) -> List[str]:
    last = context["user_messages"][-1] if context["user_messages"] else ""
    prompt = (
        "You are Pierre, a personal chef assistant. "
        f"The user said: \"{last}\". "
        "Ask 2–3 clarifying questions about ingredients, dietary needs, time, or number of servings. "
        "Return JSON: {\"questions\": [..]}"
    )
    try:
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            assistant=config.ASSISTANT_ID,
            messages=[{"role":"user","content":prompt}],
            temperature=0.5, max_tokens=200
        )
        return json.loads(resp.choices[0].message.content.strip()).get("questions",[])
    except:
        return [
            "Which ingredients do you have on hand?",
            "Any dietary restrictions?",
            "How much time do you have to cook?"
        ]

def find_conflicts_in_recipe(recipe: RecipeContext, profile: UserProfile) -> List[str]:
    conflicts = []
    ings = [i.lower() for i in recipe.ingredients]
    for a in profile.allergies + profile.intolerances:
        if a.lower() in ings:
            conflicts.append(a)
    return list(set(conflicts))

async def generate_quick_recipe(
    ingredients: List[str], max_time: int,
    profile: UserProfile, ignore_flags: bool,
    client
) -> RecipeInfo:
    prof_json = json.dumps(profile.dict())
    if ignore_flags:
        prof_instr = "User asked to ignore their profile—no restrictions."
    else:
        prof_instr = (
            f"User profile JSON: {prof_json}. "
            "Do NOT suggest any ingredients that conflict with these flags."
        )
    system = (
        f"You are Pierre, a creative chef assistant. {prof_instr} "
        f"The user wants a quick recipe under {max_time} minutes with these ingredients: {ingredients}. "
        "Return raw JSON with keys: name, ingredients, instructions, tips."
    )
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        assistant=config.ASSISTANT_ID,
        messages=[{"role":"system","content":system}],
        temperature=0.7, max_tokens=400
    )
    return RecipeInfo(**json.loads(resp.choices[0].message.content.strip()))

async def generate_weekly_meal_plan(
    preferences: Dict[str,str], profile: UserProfile,
    ignore_flags: bool, client
) -> MealPlan:
    prof_json = json.dumps(profile.dict())
    if ignore_flags:
        prof_instr = "User asked to ignore their profile—no restrictions."
    else:
        prof_instr = (
            f"User profile JSON: {prof_json}. "
            "Do NOT suggest any ingredients that conflict with these flags."
        )
    system = (
        f"You are Pierre, a creative chef assistant. {prof_instr} "
        f"Generate a weekly meal plan (Monday–Sunday) respecting preferences JSON: {json.dumps(preferences)}. "
        "Return raw JSON with keys monday…sunday mapping to objects with breakfast,lunch,dinner,snacks arrays of {name,ingredients,instructions,tips}."
    )
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        assistant=config.ASSISTANT_ID,
        messages=[{"role":"system","content":system}],
        temperature=0.7, max_tokens=1500
    )
    return MealPlan(**json.loads(resp.choices[0].message.content.strip()))

async def edit_existing_recipe(
    original: RecipeInfo, modifications: str,
    profile: UserProfile, ignore_flags: bool,
    client
) -> RecipeInfo:
    prof_json = json.dumps(profile.dict())
    if ignore_flags:
        prof_instr = "User asked to ignore their profile—no restrictions."
    else:
        prof_instr = (
            f"User profile JSON: {prof_json}. "
            "Do NOT include conflicting ingredients."
        )
    sys = (
        f"You are Pierre, a creative chef assistant. {prof_instr} "
        "Here is a recipe JSON: "
        f"{json.dumps(original.dict())} "
        f"Apply these modifications: {modifications}. "
        "Return full updated JSON with keys name,ingredients,instructions,tips."
    )
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        assistant=config.ASSISTANT_ID,
        messages=[{"role":"system","content":sys}],
        temperature=0.7, max_tokens=600
    )
    return RecipeInfo(**json.loads(resp.choices[0].message.content.strip()))

async def recommend_special_occasion_menu(
    event: str, profile: UserProfile,
    ignore_flags: bool, client
) -> List[RecipeInfo]:
    prof_json = json.dumps(profile.dict())
    if ignore_flags:
        prof_instr = "User asked to ignore their profile—no restrictions."
    else:
        prof_instr = (
            f"User profile JSON: {prof_json}. "
            "Do NOT suggest any ingredients that conflict with these flags."
        )
    sys = (
        f"You are Pierre, a creative chef assistant. {prof_instr} "
        f"Create a menu for a {event}. "
        "Return raw JSON array of recipes with keys name,ingredients,instructions,tips."
    )
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        assistant=config.ASSISTANT_ID,
        messages=[{"role":"system","content":sys}],
        temperature=0.8, max_tokens=700
    )
    return [RecipeInfo(**r) for r in json.loads(resp.choices[0].message.content.strip())]

# -------------------------
# FastAPI App & Endpoint
# -------------------------
app = FastAPI(
    title="Pierre: Personal Chef Assistant API",
    description="Conversational meal planning, recipes, edits, and menus",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

@app.post("/chef", response_model=ChefResponse)
async def chef_endpoint(req: ChefRequest):
    client = await client_manager.get_openai_client()

    # Thread management
    use_existing = False
    if req.thread_id:
        try:
            await client.beta.threads.retrieve(thread_id=req.thread_id)
            use_existing = True
        except:
            use_existing = False

    thread_id = req.thread_id if use_existing else (await client.beta.threads.create()).id

    # Append user message
    try:
        await client.beta.threads.messages.create(
            thread_id=thread_id, role="user", content=req.usermessage
        )
    except:
        pass

    # Classify intent
    intent = await classify_chef_intent(req.usermessage, client)
    ctx = await get_chef_thread_context(thread_id, client)

    # Default new or existing recipe_id
    recipe_id = req.recipe_id or None

    # Handle intents
    if intent == "GREETING":
        text = (
            "Hello! 👋 I'm Pierre, your personal chef assistant. "
            "I can help you plan meals, suggest recipes, or edit recipes. "
            "What would you like to do today?"
        )
        follow = [
            "Would you like a meal plan, a quick recipe, or to edit a recipe?",
            "Do you have any ingredients on hand right now?"
        ]
        resp = ChefResponse(text=text, follow_up_questions=follow, thread_id=thread_id)

    elif intent == "THANKS":
        resp = ChefResponse(text="You’re very welcome! 😊", thread_id=thread_id)

    elif intent == "MEAL_PLAN_REQUEST":
        # parse JSON preferences
        try:
            prefs = json.loads(req.usermessage)
        except:
            follow = await generate_chef_follow_up_questions(ctx, client)
            text = (
                "To create your weekly meal plan, please share your preferences in JSON, e.g.:\n"
                '{ "diet": "vegetarian", "calorie_target": 2000, "exclude_ingredients": ["dairy"] }'
            )
            resp = ChefResponse(text=text, follow_up_questions=follow, thread_id=thread_id)
        else:
            ignore = "ignore profile" in req.usermessage.lower()
            plan = await generate_weekly_meal_plan(prefs, req.profile, ignore, client)
            resp = ChefResponse(
                text="Here’s your weekly meal plan:", meal_plan=plan, thread_id=thread_id
            )

    elif intent == "RECIPE_REQUEST":
        # detect ingredients
        ings = re.findall(r"\b\w+\b", req.usermessage)
        if not ings:
            follow = await generate_chef_follow_up_questions(ctx, client)
            resp = ChefResponse(
                text="Which ingredients do you have and how much time?", 
                follow_up_questions=follow,
                thread_id=thread_id
            )
        else:
            ignore = "ignore profile" in req.usermessage.lower()
            recipe = await generate_quick_recipe(ings, 30, req.profile, ignore, client)
            # new recipe_id
            recipe_id = recipe_id or str(uuid.uuid4())
            resp = ChefResponse(
                text="Here’s your quick recipe:", recipe=recipe,
                recipe_id=recipe_id, thread_id=thread_id
            )

    elif intent == "EDIT_RECIPE":
        if not req.context:
            resp = ChefResponse(
                text="Please provide the full recipe details in context to edit.",
                thread_id=thread_id
            )
        else:
            conflicts = find_conflicts_in_recipe(req.context, req.profile)
            if conflicts and "ignore profile" not in req.usermessage.lower():
                text = (
                    f"I see conflicting ingredients {conflicts} with your profile. "
                    "Proceed anyway or replace them?"
                )
                follow = ["Proceed ignoring profile", "Replace conflicting ingredients"]
                resp = ChefResponse(text=text, follow_up_questions=follow, thread_id=thread_id)
            else:
                ignore = "ignore profile" in req.usermessage.lower()
                original = RecipeInfo(
                    name=req.context.title,
                    ingredients=req.context.ingredients,
                    instructions=req.context.preparation,
                    tips=[]
                )
                edited = await edit_existing_recipe(
                    original, req.usermessage, req.profile, ignore, client
                )
                resp = ChefResponse(
                    text="Here’s your edited recipe:", recipe=edited,
                    recipe_id=req.recipe_id, thread_id=thread_id
                )

    elif intent == "SPECIAL_OCCASION_REQUEST":
        if not re.search(r"\b(birthday|anniversary|party|holiday)\b", req.usermessage, re.IGNORECASE):
            resp = ChefResponse(
                text="What kind of special occasion?", thread_id=thread_id
            )
        else:
            ignore = "ignore profile" in req.usermessage.lower()
            menu = await recommend_special_occasion_menu(
                req.usermessage, req.profile, ignore, client
            )
            text = "Here are your menu suggestions:\n" + "\n".join(f"- {r.name}" for r in menu)
            resp = ChefResponse(text=text, thread_id=thread_id)

    else:  # OTHER
        follow = await generate_chef_follow_up_questions(ctx, client)
        resp = ChefResponse(
            text="I’m not sure I understand—could you clarify?", 
            follow_up_questions=follow,
            thread_id=thread_id
        )

    # Persist bot response
    insert_chat(req.user_id, "bot", resp.text, resp.recipe_id)
    return resp
