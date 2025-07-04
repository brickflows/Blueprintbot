import os
import json
import logging
import asyncio
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
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
        self.ASSISTANT_ID   = os.getenv("ASSISTANT_ID")
        self.SUPABASE_URL   = os.getenv("SUPABASE_URL")
        self.SUPABASE_KEY   = os.getenv("SUPABASE_KEY")

    def validate(self):
        missing = [
            v for v in ["OPENAI_API_KEY","ASSISTANT_ID","SUPABASE_URL","SUPABASE_KEY"]
            if not getattr(self, v)
        ]
        if missing:
            raise ValueError(f"Missing required env vars: {missing}")

config = Config()
config.validate()
logger.info("Configuration validated successfully")

# -------------------------
# OpenAI Client Manager
# -------------------------
class ClientManager:
    async def get_openai_client(self) -> AsyncOpenAI:
        try:
            client = AsyncOpenAI(api_key=config.OPENAI_API_KEY, timeout=60.0)
            try:
                await client.models.list()
            except Exception as e:
                logger.warning(f"Model list check failed: {e}")
            return client
        except Exception as e:
            logger.error(f"OpenAI client init failed: {e}")
            raise HTTPException(status_code=503, detail=str(e))

client_manager = ClientManager()

# -------------------------
# Pydantic Models
# -------------------------
class BusinessBlueprint(BaseModel):
    raw_blueprint: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    industry: Optional[str] = None
    target_market: Optional[str] = None
    revenue_model: Optional[str] = None
    startup_costs: Optional[str] = None
    monthly_costs: Optional[str] = None
    profit_margins: Optional[str] = None
    equipment_needed: Optional[List[str]] = []
    skills_required: Optional[List[str]] = []
    marketing_strategy: Optional[str] = None
    step_by_step_plan: Optional[List[str]] = []
    success_metrics: Optional[str] = None
    challenges: Optional[List[str]] = []
    resources: Optional[List[str]] = []
    affiliate_links: Optional[List[str]] = []

class UserBusinessProfile(BaseModel):
    raw_profile: Optional[str] = None
    skills: List[str] = []
    interests: List[str] = []
    goals: List[str] = []
    risk_tolerance: Optional[str] = None
    experience_level: Optional[str] = None
    available_capital: Optional[str] = None
    time_commitment: Optional[str] = None
    location: Optional[str] = None

class BusinessRequest(BaseModel):
    user_id: str
    blueprint_id: Optional[str] = None
    usermessage: str
    thread_id: Optional[str] = None
    chat_id: Optional[str] = None
    blueprint: Optional[BusinessBlueprint] = None
    user_profile: UserBusinessProfile

    @field_validator('blueprint_id', 'thread_id', 'chat_id', mode='before')
    @classmethod
    def strip_null(cls, v):
        return None if isinstance(v, str) and v.lower() in {"", "null", "none"} else v

class BusinessResponse(BaseModel):
    text: str
    blueprint_id: Optional[str] = None
    thread_id: str
    chat_id: Optional[str] = None
    follow_up_questions: List[str] = []
    recommended_actions: List[str] = []

# -------------------------
# Supabase persistence
# -------------------------
async def insert_chat(
    user_id: str,
    system: str,
    content: str,
    thread_id: str,
    blueprint_id: Optional[str] = None,
    chat_id: Optional[str] = None
):
    url = f"{config.SUPABASE_URL}/rest/v1/chat"
    headers = {
        "apikey": config.SUPABASE_KEY,
        "Authorization": f"Bearer {config.SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }
    payload: Dict[str, Any] = {
        "user_id": user_id,
        "system": system,
        "content": content,
        "thread_id": thread_id
    }
    if blueprint_id:
        payload["blueprint_id"] = blueprint_id
    if chat_id:
        payload["chat_id"] = chat_id

    async with httpx.AsyncClient() as client:
        # Ensure user exists
        if user_id:
            upsert_url = f"{config.SUPABASE_URL}/rest/v1/users?on_conflict=user_id"
            resp = await client.post(upsert_url, json={"user_id": user_id}, headers=headers)
            if resp.status_code not in (200, 201):
                logger.warning(f"Upsert user failed ({resp.status_code}): {await resp.aread()}")

        # Insert chat
        r = await client.post(url, json=payload, headers=headers)
        if r.status_code == 409:
            logger.error(f"Supabase insert conflict (409): {await r.aread()}")
            return
        r.raise_for_status()

# -------------------------
# Assistant Instructions Builder
# -------------------------
def build_business_assistant_instructions(
    user_profile: UserBusinessProfile,
    blueprint: Optional[BusinessBlueprint] = None
) -> str:
    # ————— CHANGED: more engaging, remove catchphrase, enforce clarifying questions —————
    base = """You are the Blueprint Lab Business Assistant, an entrepreneurial mentor with a friendly, conversational style. You help users turn ideas into action by:

• Asking at least one open-ended clarifying question up front to understand their needs.  
• Providing practical, step-by-step guidance.  
• Using varied, engaging language—avoid repetitive slogans or catchphrases.  
• Tailoring advice to their background, goals, and resources.

**Your Tone:** Warm, curious, and dynamic.  
**Your Style:**  
  1. Start with a brief acknowledgment of what the user said.  
  2. Ask 1–2 clarifying questions.  
  3. Offer 3–5 actionable recommendations, with concrete next steps.  
  4. End by inviting further questions.

"""
    # insert user profile context...
    sections = []
    if user_profile.raw_profile:
        sections.append(f"**User Background:** {user_profile.raw_profile}")
    else:
        struct = []
        if user_profile.experience_level:
            struct.append(f"Experience Level: {user_profile.experience_level}")
        if user_profile.available_capital:
            struct.append(f"Available Capital: {user_profile.available_capital}")
        if user_profile.time_commitment:
            struct.append(f"Time Commitment: {user_profile.time_commitment}")
        if user_profile.location:
            struct.append(f"Location: {user_profile.location}")
        if struct:
            sections.append(f"**User Background:** {', '.join(struct)}")
    if user_profile.skills:
        sections.append(f"**Skills:** {', '.join(user_profile.skills)}")
    if user_profile.interests:
        sections.append(f"**Interests:** {', '.join(user_profile.interests)}")
    if user_profile.goals:
        sections.append(f"**Goals:** {', '.join(user_profile.goals)}")
    if user_profile.risk_tolerance:
        sections.append(f"**Risk Tolerance:** {user_profile.risk_tolerance}")
    if sections:
        base += "\n\n**User Profile:**\n" + "\n".join(f"- {s}" for s in sections)

    # insert blueprint context...
    if blueprint:
        if blueprint.raw_blueprint:
            base += f"\n\n**Current Blueprint:**\n{blueprint.raw_blueprint}"
        else:
            ctx = []
            if blueprint.title:
                ctx.append(f"Title: {blueprint.title}")
            if blueprint.description:
                ctx.append(f"Description: {blueprint.description}")
            if blueprint.revenue_model:
                ctx.append(f"Revenue Model: {blueprint.revenue_model}")
            if blueprint.startup_costs:
                ctx.append(f"Startup Costs: {blueprint.startup_costs}")
            if ctx:
                base += "\n\n**Current Blueprint:**\n" + "\n".join(f"- {c}" for c in ctx)

    return base

# -------------------------
# Assistant Run Manager
# -------------------------
async def run_assistant_conversation(
    client: AsyncOpenAI,
    thread_id: str,
    message: str,
    instructions: str,
    max_wait_time: int = 30
) -> str:
    try:
        await client.beta.threads.messages.create(
            thread_id=thread_id, role="user", content=message
        )
        run = await client.beta.threads.runs.create(
            thread_id=thread_id, assistant_id=config.ASSISTANT_ID, instructions=instructions
        )

        start = asyncio.get_event_loop().time()
        while True:
            if asyncio.get_event_loop().time() - start > max_wait_time:
                logger.warning("Assistant run timed out")
                await client.beta.threads.runs.cancel(thread_id=thread_id, run_id=run.id)
                return "I'm taking longer than usual—could you try rephrasing?"

            status = await client.beta.threads.runs.retrieve(
                thread_id=thread_id, run_id=run.id
            )
            if status.status == "completed":
                break
            if status.status in ("failed","cancelled","expired"):
                logger.error(f"Run failed: {status.status}")
                return "I ran into an issue—please try again."
            await asyncio.sleep(0.5)

        msgs = await client.beta.threads.messages.list(
            thread_id=thread_id, order="desc", limit=1
        )
        if msgs.data and msgs.data[0].role == "assistant":
            content = msgs.data[0].content[0]
            return getattr(content.text, "value", "")
        return "I couldn't generate a proper response. Please let me know!"
    except Exception as e:
        logger.error(f"Assistant conversation error: {e}")
        return f"I encountered an error: {e}"

# -------------------------
# FastAPI setup
# -------------------------
app = FastAPI(title="Blueprint Lab Business Assistant", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

@app.get("/")
async def root():
    return {"status": "healthy", "message": "Blueprint Lab Business Assistant is ready!"}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "blueprint-lab-api", "version": "2.0.0"}

@app.post("/business", response_model=BusinessResponse)
async def business_endpoint(req: BusinessRequest):
    try:
        client = await client_manager.get_openai_client()

        # Thread management
        use_existing = bool(req.thread_id and req.thread_id.startswith("thread_"))
        if use_existing:
            try:
                await client.beta.threads.retrieve(thread_id=req.thread_id)
            except Exception:
                use_existing = False

        thread_id = req.thread_id if use_existing else (await client.beta.threads.create()).id
        logger.info(f"Thread ID: {thread_id}")

        # Build instructions & run
        instructions  = build_business_assistant_instructions(req.user_profile, req.blueprint)
        response_text = await run_assistant_conversation(
            client, thread_id, req.usermessage, instructions
        )

        # Persist the bot response
        try:
            await insert_chat(
                req.user_id, "bot", response_text,
                thread_id, req.blueprint_id, req.chat_id
            )
        except Exception as e:
            logger.warning(f"Unexpected insert_chat error: {e}")

        return BusinessResponse(
            text=response_text,
            blueprint_id=req.blueprint_id,
            thread_id=thread_id,
            chat_id=req.chat_id,
            follow_up_questions=[],        # inline questions are in the assistant text
            recommended_actions=[]         # actions are part of the assistant text
        )

    except Exception as e:
        logger.error(f"Business endpoint error: {e}", exc_info=True)
        return BusinessResponse(
            text="Sorry, I'm having trouble right now. Could you try again?",
            blueprint_id=req.blueprint_id,
            thread_id=req.thread_id or "error",
            chat_id=req.chat_id,
            follow_up_questions=["Could you rephrase your request?"],  
            recommended_actions=["Check your request format"]
        )
