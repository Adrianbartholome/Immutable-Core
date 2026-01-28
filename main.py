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
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# --- APP INITIALIZATION ---
app = FastAPI(title="Aether Titan Core (Glass House)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LOGGING UTILS ---
def log(message):
    print(f"[TITAN-LOG] {message}", file=sys.stdout, flush=True)

def log_error(message):
    print(f"[TITAN-ERROR] {message}", file=sys.stderr, flush=True)

# --- CONFIG CHECKS ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    log_error("CRITICAL: GEMINI_API_KEY is missing.")

try:
    GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    log_error(f"CRITICAL: Gemini Client failed to init: {e}")
    raise e

# --- DATABASE MANAGER (NO SAFETY NETS) ---
def get_db_connection_string():
    password = urllib.parse.quote_plus(os.environ.get("DB_PASSWORD", ""))
    # Force SSL mode to require to ensure we aren't getting rejected for security
    return f"postgresql://{os.environ.get('DB_USER')}:{password}@{os.environ.get('DB_HOST')}:{os.environ.get('DB_PORT', '5432')}/{os.environ.get('DB_NAME')}?sslmode=require"

class DBManager:
    def __init__(self):
        self.connection_string = get_db_connection_string()

    def connect(self):
        # We let this CRASH if it fails. No try/except.
        return psycopg2.connect(self.connection_string)

    def load_token_cache(self):
        token_cache = {}
        conn = self.connect() # Will crash here if DB is down
        with conn.cursor() as cur:
            cur.execute("SELECT english_phrase, hash_code FROM token_dictionary;")
            for p, c in cur.fetchall(): token_cache[p.strip()] = c.strip()
        conn.close()
        return token_cache

    def commit_lithograph(self, raw_text, previous_hash):
        log("Attempting Lithographic Commit...")
        # 1. Connect
        conn = self.connect()
        
        # 2. Process
        now = datetime.now()
        # Simple hash generation for debug stability
        raw_content = str(previous_hash) + raw_text
        current_hash = hashlib.sha256(raw_content.encode('utf-8')).hexdigest()
        
        # 3. Write
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chronicles (weighted_score, created_at, memory_text, previous_hash, current_hash) VALUES (%s, %s, %s, %s, %s) RETURNING id;", 
                (5, now, raw_text, previous_hash, current_hash)
            )
            new_id = cur.fetchone()[0]
        
        conn.commit()
        conn.close()
        log(f"Lithograph SUCCESS. ID: {new_id}")
        return new_id

class HolographicManager:
    def __init__(self):
        self.db = DBManager()

    def commit_hologram(self, packet, litho_id_ref=None):
        log(f"Attempting Hologram Commit for LithoID {litho_id_ref}...")
        conn = self.db.connect()
        hid = str(uuid.uuid4())
        
        # Defaults
        catalyst = packet.get('catalyst') or "System"
        mythos = packet.get('mythos') or "Observer"
        pathos = json.dumps(packet.get('pathos') or {"status": "Neutral"})
        ethos = packet.get('ethos') or "Signal"
        synthesis = packet.get('synthesis') or "Anchored"
        logos = packet.get('logos') or "Raw"

        with conn.cursor() as cur:
            cur.execute("INSERT INTO node_foundation (hologram_id, catalyst) VALUES (%s::uuid, %s)", (hid, catalyst))
            cur.execute("INSERT INTO node_essence (hologram_id, pathos, mythos) VALUES (%s::uuid, %s::jsonb, %s)", (hid, pathos, mythos))
            cur.execute("INSERT INTO node_mission (hologram_id, ethos, synthesis) VALUES (%s::uuid, %s, %s)", (hid, ethos, synthesis))
            cur.execute("INSERT INTO node_data (hologram_id, logos) VALUES (%s::uuid, %s)", (hid, logos))
            
        conn.commit()
        conn.close()
        log(f"Hologram SUCCESS. ID: {hid}")

# --- BACKGROUND WORKER ---
def background_worker(content_to_save: str, litho_id: int):
    log("BACKGROUND WORKER STARTED.")
    try:
        # 1. Refract
        log("Calling Gemini...")
        refraction = GEMINI_CLIENT.models.generate_content(
            model='gemini-2.5-flash',
            contents=[f"Return valid JSON for Aether Core based on: {content_to_save}"],
            config={"response_mime_type": "application/json"}
        )
        packet = json.loads(refraction.text)
        log("Gemini Refraction Complete.")

        # 2. Write
        hm = HolographicManager()
        hm.commit_hologram(packet, litho_id)
        
    except Exception as e:
        # THIS IS THE CRITICAL LINE.
        # If this prints, we know WHY it failed.
        log_error(f"BACKGROUND CRASH: {traceback.format_exc()}")

# --- API ---
class EventModel(BaseModel):
    action: Optional[str] = None
    query: Optional[str] = None
    memory_text: Optional[str] = None
    commit_type: Optional[str] = 'memory'

@app.post("/")
async def handle_request(event: EventModel, background_tasks: BackgroundTasks):
    log(f"Received Request: {event.commit_type}")
    
    # 1. Retrieve (Simplified)
    if event.action == 'retrieve':
        return {"status": "SUCCESS", "results": []}

    # 2. Lithograph (Sync) - THIS WILL CRASH IF DB IS BROKEN
    db = DBManager()
    litho_id = db.commit_lithograph(event.memory_text or "Heartbeat", "prev_hash_placeholder")

    # 3. Hologram (Async)
    background_tasks.add_task(background_worker, event.memory_text or "Heartbeat", litho_id)
    
    return {"status": "SUCCESS", "litho_id": litho_id, "hologram_status": "QUEUED"}

@app.get("/health")
def health():
    return {"status": "ONLINE"}