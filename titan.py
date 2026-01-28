import hashlib
import json
import os
import psycopg2
import re
import uuid
import urllib.parse
import traceback
import sys
import socket # <--- NEW IMPORT
from datetime import datetime
from google import genai
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# --- NETWORK PATCH (FORCE IPv4) ---
# This forces the app to ignore the broken IPv6 route and use the reliable IPv4 road.
old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    responses = old_getaddrinfo(*args, **kwargs)
    return [response for response in responses if response[0] == socket.AF_INET]
socket.getaddrinfo = new_getaddrinfo
# ----------------------------------

# --- APP INITIALIZATION ---
app = FastAPI(title="Aether Titan Core (IPv4 Patched)")

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
    if GEMINI_API_KEY:
        GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
        log("Gemini Client Initialized.")
    else:
        GEMINI_CLIENT = None
except Exception as e:
    log_error(f"CRITICAL: Gemini Client failed to init: {e}")
    GEMINI_CLIENT = None

# --- DATABASE MANAGER ---
def get_db_connection_string():
    password = urllib.parse.quote_plus(os.environ.get("DB_PASSWORD", ""))
    # Keep port 5432 (Session Mode) but now forcing IPv4
    return f"postgresql://{os.environ.get('DB_USER')}:{password}@{os.environ.get('DB_HOST')}:{os.environ.get('DB_PORT', '5432')}/{os.environ.get('DB_NAME')}?sslmode=require"

class DBManager:
    def __init__(self):
        self.connection_string = get_db_connection_string()

    def connect(self):
        # This will now use IPv4 thanks to the patch above
        return psycopg2.connect(self.connection_string)

    def load_token_cache(self):
        token_cache = {}
        try:
            conn = self.connect()
            with conn.cursor() as cur:
                cur.execute("SELECT english_phrase, hash_code FROM token_dictionary;")
                for p, c in cur.fetchall(): token_cache[p.strip()] = c.strip()
            conn.close()
            return token_cache
        except Exception as e:
            log_error(f"Cache Load Error: {e}")
            return {}

    def commit_lithograph(self, raw_text, previous_hash, score=5):
        log("Attempting Lithographic Commit...")
        conn = self.connect()
        
        now = datetime.now()
        raw_content = str(previous_hash) + raw_text + str(score)
        current_hash = hashlib.sha256(raw_content.encode('utf-8')).hexdigest()
        
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chronicles (weighted_score, created_at, memory_text, previous_hash, current_hash) VALUES (%s, %s, %s, %s, %s) RETURNING id;", 
                (score, now, raw_text, previous_hash, current_hash)
            )
            new_id = cur.fetchone()[0]
        
        conn.commit()
        conn.close()
        log(f"Lithograph SUCCESS. ID: {new_id}")
        return {"id": new_id, "hash": current_hash}

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
        if not GEMINI_CLIENT:
            log_error("Gemini Client missing. Skipping Refraction.")
            return

        # 1. Refract
        log("Calling Gemini for Refraction...")
        REFRACTOR_PROMPT = """You are the Aether Prism. Refract the input into JSON.
        Keys: chronos, logos, pathos, ethos, mythos, catalyst, synthesis."""
        
        refraction = GEMINI_CLIENT.models.generate_content(
            model='gemini-2.5-flash',
            contents=[REFRACTOR_PROMPT + f"\nINPUT: {content_to_save}"],
            config={"response_mime_type": "application/json"}
        )
        packet = json.loads(refraction.text)
        log("Gemini Refraction Complete.")

        # 2. Write
        hm = HolographicManager()
        hm.commit_hologram(packet, litho_id)
        
    except Exception as e:
        log_error(f"BACKGROUND CRASH: {traceback.format_exc()}")

# --- API ---
class EventModel(BaseModel):
    action: Optional[str] = None
    query: Optional[str] = None
    memory_text: Optional[str] = None
    commit_type: Optional[str] = 'memory'

@app.get("/")
def root_health_check():
    return {"status": "TITAN ONLINE", "version": "IPv4_PATCHED"}

@app.post("/")
async def handle_request(event: EventModel, background_tasks: BackgroundTasks):
    log(f"Received Request: {event.commit_type}")
    
    if event.action == 'retrieve':
        return {"status": "SUCCESS", "results": []}

    # 1. Summarize if needed (Server Side)
    final_text = event.memory_text or "Heartbeat"
    if event.commit_type == 'summary' and GEMINI_CLIENT:
        try:
            log("Generating Summary...")
            summary = GEMINI_CLIENT.models.generate_content(
                model='gemini-2.5-flash',
                contents=[f"Summarize concisely: {event.memory_text}"]
            )
            final_text = summary.text
        except Exception as e:
            log_error(f"Summary Failed: {e}")

    # 2. Lithograph (Sync)
    db = DBManager()
    # We pass a placeholder hash for now to keep it simple
    litho_res = db.commit_lithograph(final_text, "prev_hash_placeholder")

    # 3. Hologram (Async)
    background_tasks.add_task(background_worker, final_text, litho_res['id'])
    
    return {"status": "SUCCESS", "litho_id": litho_res['id'], "hologram_status": "QUEUED"}