import os
import json
import re
import uuid
import logging
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from openai import AsyncOpenAI
import httpx

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
        self.ASSISTANT_ID    = os.getenv("ASSISTANT_ID")
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
# OpenAI Client Manager
# -------------------------
class ClientManager:
    async def get_openai_client(self) -> AsyncOpenAI:
        try:
            client = AsyncOpenAI(
                api_key=config.OPENAI_API_KEY,
                timeout=30.0
            )
            try:
                await client.models.list()
            except Exception as model_error:
                logger.warning(f"Model list check failed: {model_error}")
            return client
        except Exception as e:
            logger.error(f"Failed to create OpenAI client: {e}")
            raise HTTPException(status_code=503, detail=f"OpenAI connection failed: {e}")

client_manager = ClientManager()

# -------------------------
# Helper functions to parse stringified recipe context
# -------------------------
def extract_ingredient_list(s: str) -> List[str]:
    blocks = re.findall(r"\{([^}]+)\}", s)
    out = []
    for block in blocks:
        m = re.search(r"ingredient\s*:\s*([^,}]+)", block)
        if m:
            out.append(m.group(1).strip())
    return out

def extract_instruction_list(s: str) -> List[str]:
    blocks = re.findall(r"\{([^}]+)\}", s)
    out = []
    for block in blocks:
        m = re.search(r"instruction\s*:\s*([^,}]+(?:,.*)?)$", block)
        if m:
            out.append(m.group(1).strip())
    return out

# -------------------------
# Pydantic Models
# -------------------------
class UserProfile(BaseModel):
    dietary_preferences: List[str] = []
    allergies:             List[str] = []
    intolerances:          List[str] = []
    other_flags:           List[str] = []

class RecipeContext(BaseModel):
    title:             Optional[str] = None
    description:       Optional[str] = None
    ingredients:       Optional[Any] = None
    preparation:       Optional[Any] = None
    nutrition_content: Optional[str] = None

    @field_validator('ingredients', mode='before')
    @classmethod
    def parse_ingredients(cls, v):
        if isinstance(v, list):
            return [item.get("ingredient", str(item)) for item in v]
        if isinstance(v, str):
            try:
                arr = json.loads(v)
                return [d.get("ingredient", str(d)) for d in arr]
            except Exception:
                return extract_ingredient_list(v)
        return None

    @field_validator('preparation', mode='before')
    @classmethod
    def parse_preparation(cls, v):
        if isinstance(v, list):
            return [item.get("instruction", str(item)) for item in v]
        if isinstance(v, str):
            try:
                arr = json.loads(v)
                return [d.get("instruction", str(d)) for d in arr]
            except Exception:
                return extract_instruction_list(v)
        return None

class ChefRequest(BaseModel):
    user_id:     str
    recipe_id:   Optional[str]         = None
    usermessage: str
    thread_id:   Optional[str]         = None
    context:     Optional[RecipeContext] = None
    profile:     UserProfile

    @field_validator('recipe_id', 'thread_id', 'context', mode='before')
    @classmethod
    def validate_null_strings(cls, v):
        if isinstance(v, str) and v.lower() in ['null', 'none', ''] or 'null' in str(v).lower():
            return None
        return v

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
# Supabase chat persistence
# -------------------------
async def insert_chat(
    user_id: str,
    system_role: str,
    content: str,
    thread_id: str,
    recipe_id: Optional[str] = None
):
    url = f"{config.SUPABASE_URL}/rest/v1/chat"
    headers = {
        "apikey": config.SUPABASE_KEY,
        "Authorization": f"Bearer {config.SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }
    payload = {
        "user_id": user_id,
        "system":  system_role,
        "content": content,
        "thread_id": thread_id,
    }
    if recipe_id:
        payload["recipe_id"] = recipe_id

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()

# -------------------------
# OpenAI / Chef logic helpers
# -------------------------
async def get_chef_thread_context(thread_id: str, client) -> Dict:
    try:
        msgs = await client.beta.threads.messages.list(
            thread_id=thread_id, order="asc", limit=100
        )
        user_msgs = []
        for m in msgs.data:
            if m.role == "user" and m.content:
                text = getattr(m.content[0].text, "value", None)
                if text:
                    user_msgs.append(text)
        return {"user_messages": user_msgs}
    except Exception as e:
        logger.warning(f"Failed to fetch thread context: {e}")
        return {"user_messages": []}

async def classify_chef_intent(message: str, client) -> str:
    prompt = (
        "You are a classifier for a cooking assistant. Label the user message "
        "with exactly one of: GREETING, THANKS, RECIPE_REQUEST, "
        "MEAL_PLAN_REQUEST, EDIT_RECIPE, SPECIAL_OCCASION_REQUEST, OTHER.\n\n"
        f"User: \"{message.strip()}\""
    )
    try:
        resp = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20
        )
        label = resp.choices[0].message.content.strip().upper()
        valid = {
            "GREETING","THANKS","RECIPE_REQUEST","MEAL_PLAN_REQUEST",
            "EDIT_RECIPE","SPECIAL_OCCASION_REQUEST","OTHER"
        }
        if label in valid:
            return label
    except Exception as e:
        logger.warning(f"Intent classification failed: {e}")

    # fallback keyword check
    low = message.lower()
    if any(k in low for k in ['replace', 'substitute', 'change', 'modify']):
        return "EDIT_RECIPE"
    return "OTHER"

async def generate_chef_follow_up_questions(context: Dict, client) -> List[str]:
    last = context["user_messages"][-1] if context["user_messages"] else ""
    prompt = (
        f"You are Pierre, a personal chef assistant. The user said: \"{last}\". "
        "Ask 2–3 clarifying questions about ingredients, dietary needs, time, or servings. "
        "Return JSON: {\"questions\": [...]}"
    )
    try:
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.5, max_tokens=200
        )
        return json.loads(resp.choices[0].message.content.strip()).get("questions", [])
    except Exception:
        return [
            "Which ingredients do you have on hand?",
            "Any dietary restrictions?",
            "How much time do you have to cook?"
        ]

def find_conflicts_in_recipe(recipe: RecipeContext, profile: UserProfile) -> List[str]:
    conflicts = []
    if recipe.ingredients:
        text = " ".join(recipe.ingredients).lower()
        for flag in profile.allergies + profile.intolerances:
            if flag.lower() in text:
                conflicts.append(flag)
    return list(set(conflicts))

async def generate_quick_recipe(
    ingredients: List[str],
    max_time: int,
    profile: UserProfile,
    ignore_flags: bool,
    client
) -> RecipeInfo:
    prof_json = json.dumps(profile.dict())
    prof_instr = "User asked to ignore their profile—no restrictions." if ignore_flags else (
        f"User profile JSON: {prof_json}. Do NOT suggest any ingredients that conflict with these flags."
    )
    system = (
        f"You are Pierre. {prof_instr} "
        f"The user wants a quick recipe under {max_time} minutes using: {ingredients}. "
        "Return JSON with keys: name, ingredients, instructions, tips."
    )
    try:
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":system}],
            temperature=0.7, max_tokens=400
        )
        return RecipeInfo(**json.loads(resp.choices[0].message.content.strip()))
    except Exception as e:
        logger.error(f"Recipe generation failed: {e}")
        return RecipeInfo(
            name="Simple Recipe",
            ingredients=ingredients[:5],
            instructions=["Combine ingredients", "Cook as desired"],
            tips=["Adjust seasoning to taste"]
        )

async def generate_weekly_meal_plan(
    preferences: Dict[str, Any],
    profile: UserProfile,
    ignore_flags: bool,
    client
) -> MealPlan:
    prof_json = json.dumps(profile.dict())
    prof_instr = "User asked to ignore their profile—no restrictions." if ignore_flags else (
        f"User profile JSON: {prof_json}. Do NOT suggest any ingredients that conflict with these flags."
    )
    system = (
        f"You are Pierre. {prof_instr} "
        f"Generate a weekly meal plan (Mon–Sun) respecting preferences: {json.dumps(preferences)}. "
        "Return raw JSON mapping monday…sunday to objects with breakfast,lunch,dinner,snacks arrays."
    )
    try:
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":system}],
            temperature=0.7, max_tokens=1500
        )
        return MealPlan(**json.loads(resp.choices[0].message.content.strip()))
    except Exception as e:
        logger.error(f"Meal plan generation failed: {e}")
        simple = RecipeInfo(name="Simple Meal", ingredients=["basic ingredients"], instructions=["Prepare as desired"], tips=[])
        day = MealPlanDay(breakfast=[simple], lunch=[simple], dinner=[simple], snacks=[])
        return MealPlan(**{d: day for d in ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]})

async def edit_existing_recipe(
    original: RecipeInfo,
    modifications: str,
    profile: UserProfile,
    ignore_flags: bool,
    client
) -> RecipeInfo:
    prof_json = json.dumps(profile.dict())
    prof_instr = "User asked to ignore their profile—no restrictions." if ignore_flags else (
        f"User profile JSON: {prof_json}. Do NOT include conflicting ingredients."
    )
    system = (
        f"You are Pierre. {prof_instr} "
        f"Here is a recipe JSON: {json.dumps(original.dict())} "
        f"Apply these modifications: {modifications}. "
        "Return full updated JSON with keys name,ingredients,instructions,tips."
    )
    try:
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":system}],
            temperature=0.7, max_tokens=600
        )
        return RecipeInfo(**json.loads(resp.choices[0].message.content.strip()))
    except Exception as e:
        logger.error(f"Recipe editing failed: {e}")
        return original

async def recommend_special_occasion_menu(
    event: str,
    profile: UserProfile,
    ignore_flags: bool,
    client
) -> List[RecipeInfo]:
    prof_json = json.dumps(profile.dict())
    prof_instr = "User asked to ignore their profile—no restrictions." if ignore_flags else (
        f"User profile JSON: {prof_json}. Do NOT suggest any ingredients that conflict with these flags."
    )
    system = (
        f"You are Pierre. {prof_instr} "
        f"Create a menu for a {event}. "
        "Return JSON array of recipes with keys name,ingredients,instructions,tips."
    )
    try:
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":system}],
            temperature=0.8, max_tokens=700
        )
        return [RecipeInfo(**r) for r in json.loads(resp.choices[0].message.content.strip())]
    except Exception as e:
        logger.error(f"Special occasion menu failed: {e}")
        return [RecipeInfo(name="Special Dish", ingredients=["seasonal ingredients"], instructions=["Prepare with care"], tips=["Make it special"])]

# -------------------------
# FastAPI app & endpoints
# -------------------------
app = FastAPI(
    title="Pierre: Personal Chef Assistant API",
    description="Conversational meal planning, recipes, edits, and menus",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
async def root():
    return {"message": "Pierre Chef Assistant API is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "pierre-chef-api"}

@app.post("/chef", response_model=ChefResponse)
async def chef_endpoint(req: ChefRequest):
    try:
        client = await client_manager.get_openai_client()

        # Thread management
        use_existing = False
        if req.thread_id and req.thread_id.startswith("thread_"):
            try:
                await client.beta.threads.retrieve(thread_id=req.thread_id)
                use_existing = True
            except Exception as e:
                logger.warning(f"Thread retrieval failed: {e}")

        thread_id = req.thread_id if use_existing else (await client.beta.threads.create()).id

        # Append user message
        try:
            await client.beta.threads.messages.create(
                thread_id=thread_id, role="user", content=req.usermessage
            )
        except Exception as e:
            logger.warning(f"Message creation failed: {e}")

        intent = await classify_chef_intent(req.usermessage, client)
        ctx = await get_chef_thread_context(thread_id, client)
        recipe_id = req.recipe_id

        # Handle intents
        if intent == "GREETING":
            text = (
                "Hello! 👋 I'm Pierre, your personal chef assistant. "
                "I can help with meal plans, recipes, or editing recipes. "
                "What would you like to do today?"
            )
            follow = [
                "Would you like a meal plan, a quick recipe, or to edit a recipe?",
                "Do you have any ingredients on hand right now?"
            ]
            resp = ChefResponse(text=text, follow_up_questions=follow, thread_id=thread_id)

        elif intent == "THANKS":
            resp = ChefResponse(text="You're very welcome! 😊", thread_id=thread_id)

        elif intent == "MEAL_PLAN_REQUEST":
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
                    text="Here's your weekly meal plan:", meal_plan=plan, thread_id=thread_id
                )

        elif intent == "RECIPE_REQUEST":
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
                recipe_id = recipe_id or str(uuid.uuid4())
                resp = ChefResponse(
                    text="Here's your quick recipe:", recipe=recipe,
                    recipe_id=recipe_id, thread_id=thread_id
                )

        elif intent == "EDIT_RECIPE":
            # require complete context
            if not req.context \
               or not req.context.title \
               or not req.context.ingredients \
               or not req.context.preparation:
                return ChefResponse(
                    text=(
                        "To edit a recipe, please send me the original recipe in your `context` as JSON, e.g.:\n"
                        "```json\n"
                        "{\n"
                        '  "context": {\n'
                        '    "title": "Pancakes",\n'
                        '    "ingredients": ["flour","milk","egg"],\n'
                        '    "preparation": ["Mix ingredients","Cook on skillet"]\n'
                        "  }\n"
                        "}\n"
                        "```"
                    ),
                    thread_id=thread_id
                )

            conflicts = find_conflicts_in_recipe(req.context, req.profile)
            if conflicts and "ignore profile" not in req.usermessage.lower():
                text = (
                    f"I see conflicting ingredients {conflicts}. "
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
                    text="Here's your edited recipe:",
                    recipe=edited,
                    recipe_id=req.recipe_id,
                    thread_id=thread_id
                )

        elif intent == "SPECIAL_OCCASION_REQUEST":
            if not re.search(r"\b(birthday|anniversary|party|holiday)\b", req.usermessage, re.IGNORECASE):
                resp = ChefResponse(text="What kind of special occasion?", thread_id=thread_id)
            else:
                ignore = "ignore profile" in req.usermessage.lower()
                menu = await recommend_special_occasion_menu(
                    req.usermessage, req.profile, ignore, client
                )
                text = "Here are your menu suggestions:\n" + "\n".join(f"- {r.name}" for r in menu)
                resp = ChefResponse(text=text, thread_id=thread_id)

        else:
            follow = await generate_chef_follow_up_questions(ctx, client)
            resp = ChefResponse(
                text="I'm not sure I understand—could you clarify?",
                follow_up_questions=follow,
                thread_id=thread_id
            )

        # Persist bot response
        try:
            await insert_chat(req.user_id, "bot", resp.text, resp.thread_id, resp.recipe_id)
        except Exception as e:
            logger.warning(f"Chat persistence failed: {e}")

        return resp

    except Exception as e:
        logger.error(f"Chef endpoint error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return ChefResponse(
            text=f"I encountered an error: {e}. Please try a simpler request.",
            thread_id=str(uuid.uuid4())
        )
