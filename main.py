import hashlib
import json
import os
import psycopg2
import re
import uuid
import urllib.parse
import traceback
import sys
from datetime import datetime
from google import genai
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any

# --- APP INITIALIZATION ---
app = FastAPI(title="Aether Titan Core")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- UTILS ---
def log(message):
    """Force flush logs so they show up in DigitalOcean immediately"""
    print(f"[TITAN-LOG] {message}", file=sys.stdout, flush=True)

def log_error(message):
    print(f"[TITAN-ERROR] {message}", file=sys.stderr, flush=True)

# --- GLOBAL VARIABLES ---
TOKEN_DICTIONARY_CACHE = {}
GEMINI_CLIENT = None 

# --- SECURE CLIENT INITIALIZATION ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    log_error("FATAL: GEMINI_API_KEY not found.")
else:
    try:
        GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
        log("Gemini Client Initialized.")
    except Exception as e:
        log_error(f"FATAL: Gemini Init Failed: {e}")

# --- DB MANAGER ---
def get_db_connection_string():
    password = urllib.parse.quote_plus(os.environ.get("DB_PASSWORD", ""))
    return f"postgresql://{os.environ.get('DB_USER')}:{password}@{os.environ.get('DB_HOST')}:{os.environ.get('DB_PORT', '6543')}/{os.environ.get('DB_NAME')}?sslmode=require"

class DBManager:
    def __init__(self):
        self.connection_string = get_db_connection_string()

    def connect(self):
        return psycopg2.connect(self.connection_string)

    def search_lithograph(self, query, cache):
        # Placeholder for search logic
        return []

# --- HOLOGRAPHIC MANAGER ---
class HolographicManager:
    def __init__(self):
        self.db = DBManager()

    def commit_hologram(self, packet, litho_id_ref=None):
        hid = str(uuid.uuid4())
        conn = None
        try:
            log(f"Attempting to connect to DB for Hologram {hid}...")
            conn = self.db.connect()
            
            # Defaults
            catalyst = packet.get('catalyst') or "Implicit System Trigger"
            mythos = packet.get('mythos') or "The Observer"
            pathos = json.dumps(packet.get('pathos') or {"status": "Neutral"}) 
            ethos = packet.get('ethos') or "Preservation of Signal"
            synthesis = packet.get('synthesis') or "Data Anchored"
            logos = packet.get('logos') or "Raw Data Artifact"

            with conn.cursor() as cur:
                log(f"Executing SQL Insert for {hid}...")
                cur.execute("INSERT INTO node_foundation (hologram_id, catalyst) VALUES (%s::uuid, %s)", (hid, catalyst))
                cur.execute("INSERT INTO node_essence (hologram_id, pathos, mythos) VALUES (%s::uuid, %s::jsonb, %s)", (hid, pathos, mythos))
                cur.execute("INSERT INTO node_mission (hologram_id, ethos, synthesis) VALUES (%s::uuid, %s, %s)", (hid, ethos, synthesis))
                cur.execute("INSERT INTO node_data (hologram_id, logos) VALUES (%s::uuid, %s)", (hid, logos))
                
            conn.commit()
            log(f"SUCCESS: Hologram {hid} committed.")
            return {"status": "SUCCESS", "hologram_id": hid}

        except Exception as e:
            if conn: conn.rollback()
            log_error(f"DB WRITE FAILED: {traceback.format_exc()}")
            return {"status": "FAILURE", "error": str(e)}
        finally:
            if conn: conn.close()

# --- BACKGROUND WORKER ---
def background_hologram_process(content_to_save: str, litho_id: int):
    log(f"Background Task Started for Litho ID: {litho_id}")
    try:
        if not GEMINI_CLIENT:
            log_error("Gemini Client is None. Aborting refraction.")
            return

        # 1. Refract
        log("Calling Gemini for refraction...")
        # (Shortened prompt for debug speed)
        REFRACTOR_PROMPT = """You are the Aether Prism. Return valid JSON only.
        {"chronos": "now", "logos": "summary", "pathos": {"state": "neutral"}, "ethos": "intent", "mythos": "archetype", "catalyst": "trigger", "synthesis": "lesson"}"""
        
        refraction = GEMINI_CLIENT.models.generate_content(
            model='gemini-2.5-flash',
            contents=[REFRACTOR_PROMPT + f"\nINPUT: {content_to_save}"],
            config={"temperature": 0.1, "response_mime_type": "application/json"}
        )
        
        # Clean response
        raw_text = refraction.text.strip()
        if raw_text.startswith("```"): raw_text = raw_text.split("\n", 1)[-1].rsplit("\n", 1)[0]
        if raw_text.startswith("json"): raw_text = raw_text[4:].strip()
        
        packet = json.loads(raw_text)
        log("Refraction Complete. Packet parsed.")

        # 2. Commit
        holo_manager = HolographicManager()
        holo_manager.commit_hologram(packet, litho_id)
        
    except Exception as e:
        log_error(f"BACKGROUND CRASH: {traceback.format_exc()}")

# --- DATA MODELS ---
class EventModel(BaseModel):
    action: Optional[str] = None
    query: Optional[str] = None
    commit_type: Optional[str] = 'memory'
    memory_text: Optional[str] = None
    override_score: Optional[int] = None

# --- DEBUG ROUTES ---

@app.get("/debug/db")
def debug_db_sync():
    """TEST 1: Can we write to the DB at all?"""
    try:
        log("DEBUG: Starting Sync DB Test...")
        hm = HolographicManager()
        # Hardcoded packet
        packet = {
            "catalyst": "DEBUG_SYNC_TEST",
            "logos": "If you see this, FastAPI can talk to Postgres.",
            "pathos": {"status": "alive"},
            "mythos": "The Debugger",
            "ethos": "Sanity Check",
            "synthesis": "Connectivity Confirmed"
        }
        res = hm.commit_hologram(packet)
        return res
    except Exception as e:
        return {"status": "FAILURE", "error": str(e)}

@app.get("/debug/background")
def debug_background(background_tasks: BackgroundTasks):
    """TEST 2: Do background tasks actually run?"""
    log("DEBUG: Queuing Background Task...")
    
    def dummy_task():
        log("DEBUG: Dummy Background Task Running...")
        hm = HolographicManager()
        packet = {
            "catalyst": "DEBUG_BACKGROUND_TEST",
            "logos": "If you see this, BackgroundTasks are working.",
            "pathos": {"status": "async_alive"},
            "mythos": "The Ghost",
            "ethos": "Async Check",
            "synthesis": "Threading Confirmed"
        }
        hm.commit_hologram(packet)
    
    background_tasks.add_task(dummy_task)
    return {"status": "QUEUED", "message": "Check logs/DB for DEBUG_BACKGROUND_TEST"}

# --- MAIN ROUTE ---
@app.post("/")
async def handle_request(event: EventModel, background_tasks: BackgroundTasks):
    # (Simplified for brevity - assumes logic is mostly correct)
    if event.action == 'retrieve':
        return {"status": "SUCCESS", "results": []}

    log(f"Received Request: {event.commit_type}")
    
    # Queue the background task
    background_tasks.add_task(background_hologram_process, event.memory_text or "Empty", 0)
    
    return {"status": "SUCCESS", "hologram_status": "QUEUED_IN_BACKGROUND"}

# --- HEALTH ---
@app.get("/health")
def health():
    return {"status": "ONLINE"}