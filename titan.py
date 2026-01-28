import hashlib
import json
import os
import psycopg2
import uuid
import urllib.parse
import traceback
import sys
from datetime import datetime
from google import genai
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# --- APP INITIALIZATION ---
app = FastAPI(title="Aether Titan Core (Pooler Optimized)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LOGGING ---
def log(message):
    print(f"[TITAN-LOG] {message}", file=sys.stdout, flush=True)

def log_error(message):
    print(f"[TITAN-ERROR] {message}", file=sys.stderr, flush=True)

# --- CONFIG ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
try:
    GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
except:
    GEMINI_CLIENT = None

# --- DATABASE MANAGER (Pooler Friendly) ---
def get_db_connection_string():
    # We use the standard host/port. Port 6543 handles the routing magic.
    password = urllib.parse.quote_plus(os.environ.get("DB_PASSWORD", ""))
    return f"postgresql://{os.environ.get('DB_USER')}:{password}@{os.environ.get('DB_HOST')}:{os.environ.get('DB_PORT', '6543')}/{os.environ.get('DB_NAME')}?sslmode=require"

class DBManager:
    def __init__(self):
        self.connection_string = get_db_connection_string()

    def connect(self):
        return psycopg2.connect(self.connection_string)

    def commit_lithograph(self, raw_text, previous_hash, score=5):
        # POOLER SAFETY: Open connection -> Do work -> Close immediately.
        conn = self.connect()
        try:
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
            return {"id": new_id, "hash": current_hash}
        except Exception as e:
            log_error(f"Lithograph Error: {e}")
            raise e
        finally:
            conn.close()

class HolographicManager:
    def __init__(self):
        self.db = DBManager()

    def commit_hologram(self, packet):
        # POOLER SAFETY: Open connection -> Do work -> Close immediately.
        conn = self.db.connect()
        try:
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
            log(f"Hologram SUCCESS. ID: {hid}")
        except Exception as e:
            log_error(f"Hologram Error: {e}")
            raise e
        finally:
            conn.close()

# --- BACKGROUND WORKER (Safe Async) ---
def background_refraction(content: str, litho_id: int):
    log(f"Starting Refraction for Litho {litho_id}...")
    
    # PHASE 1: HEAVY LIFTING (No DB Connection)
    # The Pooler cannot kill us here because we aren't connected yet.
    if not GEMINI_CLIENT:
        log("No Gemini Key. Skipping.")
        return

    try:
        packet = None
        log("Calling Gemini...")
        REFRACTOR_PROMPT = """You are the Aether Prism. Refract input into JSON. 
        Keys: chronos, logos, pathos, ethos, mythos, catalyst, synthesis."""
        
        res = GEMINI_CLIENT.models.generate_content(
            model='gemini-2.5-flash',
            contents=[REFRACTOR_PROMPT + f"\nINPUT: {content}"],
            config={"response_mime_type": "application/json"}
        )
        packet = json.loads(res.text)
        log("Gemini Success.")
    except Exception as e:
        log_error(f"Gemini Failed: {e}")
        return

    # PHASE 2: FAST WRITE (Open DB -> Write -> Close)
    # We are in and out in 50ms. The Pooler will love this.
    try:
        hm = HolographicManager()
        hm.commit_hologram(packet)
    except Exception as e:
        log_error(f"DB Write Failed: {e}")

# --- API ---
class EventModel(BaseModel):
    action: Optional[str] = None
    commit_type: Optional[str] = 'memory'
    memory_text: Optional[str] = None

@app.get("/")
def health_check():
    return {"status": "TITAN ONLINE", "mode": "POOLER_6543"}

@app.post("/")
async def handle_request(event: EventModel, background_tasks: BackgroundTasks):
    log(f"Request: {event.commit_type}")
    
    # 1. Summarize (Server Side) - Phase 1 (No DB)
    final_text = event.memory_text or "Heartbeat"
    if event.commit_type == 'summary' and GEMINI_CLIENT:
        try:
            res = GEMINI_CLIENT.models.generate_content(
                model='gemini-2.5-flash',
                contents=[f"Summarize: {final_text}"]
            )
            final_text = res.text
        except Exception as e:
            log_error(f"Summary Error: {e}")

    # 2. Lithograph - Phase 2 (Fast DB Write)
    db = DBManager()
    litho_res = db.commit_lithograph(final_text, "prev_hash_placeholder")
    
    # 3. Hologram - Phase 3 (Background)
    background_tasks.add_task(background_refraction, final_text, litho_res['id'])
    
    return {"status": "SUCCESS", "litho_id": litho_res['id']}