import hashlib
import json
import os
import psycopg2
import re
import uuid
import urllib.parse
import traceback
import sys
import requests # --- NEW: FOR WEB SCRAPING
from datetime import datetime
from google import genai
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# --- APP INITIALIZATION ---
app = FastAPI(title="Aether Titan Core (Platinum V5.2 - Spider Module)")

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

# --- GLOBAL VARIABLES & PROMPTS ---
TOKEN_DICTIONARY_CACHE = {}

SCORING_SYSTEM_PROMPT = """
You are SNEGO-P, the Aether Eternal Cognitive Assessor.
Output MUST be a single integer from 0 to 9, preceded strictly by 'SCORE: '. 
Example: 'SCORE: 9'. No other text.
"""

REFRACTOR_SYSTEM_PROMPT = """
You are the Aether Prism. Refract the input into 7 channels for the Holographic Core.
Return ONLY a JSON object with these exact keys:
{
  "chronos": "ISO Timestamp",
  "logos": "The core factual text/summary",
  "pathos": {"emotion_name": score, ...},
  "ethos": "The strategic goal/intent",
  "mythos": "The active archetype",
  "catalyst": "The trigger",
  "synthesis": "The outcome/lesson"
}
"""

# --- CLIENT INITIALIZATION ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
try:
    if GEMINI_API_KEY:
        GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
        log("Gemini Client Initialized.")
    else:
        GEMINI_CLIENT = None
        log_error("GEMINI_API_KEY missing. AI features disabled.")
except Exception as e:
    log_error(f"Gemini Init Failed: {e}")
    GEMINI_CLIENT = None

# --- UTILITIES ---
def generate_hash(memory_data, previous_hash_string):
    if isinstance(memory_data.get("timestamp"), datetime):
        memory_data["timestamp"] = memory_data["timestamp"].isoformat()
    
    data_block_string = json.dumps(memory_data, sort_keys=True)
    raw_content = previous_hash_string + data_block_string
    return hashlib.sha256(raw_content.encode('utf-8')).hexdigest()

def decode_memory(compressed_text, token_map):
    if not token_map: return compressed_text
    decompressed_text = compressed_text
    decode_map = {v: k for k, v in token_map.items()}
    sorted_hashes = sorted(decode_map.keys(), key=len, reverse=True)
    for hash_code in sorted_hashes:
        if hash_code in decompressed_text:
            decompressed_text = decompressed_text.replace(hash_code, decode_map[hash_code])
    return decompressed_text

def encode_memory(raw_text, token_map):
    if not token_map: return raw_text
    compressed_text = raw_text
    sorted_tokens = sorted(token_map.items(), key=lambda item: len(item[0]), reverse=True)
    for phrase, hash_code in sorted_tokens:
        if phrase in compressed_text:
            compressed_text = compressed_text.replace(phrase, hash_code)
    return compressed_text

# --- DATABASE MANAGER ---
def get_db_connection_string():
    password = urllib.parse.quote_plus(os.environ.get("DB_PASSWORD", ""))
    return f"postgresql://{os.environ.get('DB_USER')}:{password}@{os.environ.get('DB_HOST')}:{os.environ.get('DB_PORT', '6543')}/{os.environ.get('DB_NAME')}?sslmode=require"

class DBManager:
    def __init__(self):
        self.connection_string = get_db_connection_string()

    def connect(self):
        return psycopg2.connect(self.connection_string)

    def load_token_cache(self):
        token_cache = {}
        conn = None
        try:
            conn = self.connect()
            with conn.cursor() as cur:
                cur.execute("SELECT english_phrase, hash_code FROM token_dictionary;")
                for p, c in cur.fetchall(): token_cache[p.strip()] = c.strip()
            return token_cache
        except Exception as e:
            log_error(f"Cache Load Error: {e}")
            return {}
        finally: 
            if conn: conn.close()

    def commit_lithograph(self, previous_hash, raw_text, client, token_cache, manual_score=None):
        conn = None
        try:
            compressed = encode_memory(raw_text, token_cache)
            score = 5
            if manual_score: 
                score = int(manual_score)
            
            conn = self.connect()
            now = datetime.now()
            current_hash = generate_hash({"timestamp": now, "weighted_score": score, "memory_text": compressed}, previous_hash)
            
            with conn.cursor() as cur:
                cur.execute("INSERT INTO chronicles (weighted_score, created_at, memory_text, previous_hash, current_hash, is_active) VALUES (%s, %s, %s, %s, %s, TRUE) RETURNING id;", 
                            (score, now, compressed, previous_hash, current_hash))
                new_id = cur.fetchone()[0]
            conn.commit()
            log(f"Lithograph Committed. ID: {new_id}")
            return {"status": "SUCCESS", "score": score, "new_hash": current_hash, "litho_id": new_id}
        except Exception as e:
            if conn: conn.rollback()
            log_error(f"Litho Commit Failed: {e}")
            return {"status": "FAILURE", "error": str(e)}
        finally:
            if conn: conn.close()

    def delete_lithograph(self, target_id):
        conn = None
        try:
            conn = self.connect()
            with conn.cursor() as cur:
                cur.execute("UPDATE chronicles SET is_active = FALSE WHERE id = %s RETURNING id;", (target_id,))
                if cur.rowcount == 0:
                    return {"status": "FAILURE", "error": "ID not found"}
                deleted_id = cur.fetchone()[0]
            conn.commit()
            log(f"Lithograph {deleted_id} DEACTIVATED.")
            return {"status": "SUCCESS", "deleted_id": deleted_id}
        except Exception as e:
            if conn: conn.rollback()
            log_error(f"Delete Error: {e}")
            return {"status": "FAILURE", "error": str(e)}
        finally:
            if conn: conn.close()

    def delete_range(self, start_id, end_id):
        conn = None
        try:
            conn = self.connect()
            with conn.cursor() as cur:
                cur.execute("UPDATE chronicles SET is_active = FALSE WHERE id >= %s AND id <= %s RETURNING id;", (start_id, end_id))
                count = len(cur.fetchall())
            conn.commit()
            log(f"BATCH DEACTIVATION: IDs {start_id} to {end_id} ({count} records).")
            return {"status": "SUCCESS", "deleted_count": count}
        except Exception as e:
            if conn: conn.rollback()
            log_error(f"Batch Delete Error: {e}")
            return {"status": "FAILURE", "error": str(e)}
        finally:
            if conn: conn.close()

    def restore_range(self, start_id, end_id):
        conn = None
        try:
            conn = self.connect()
            with conn.cursor() as cur:
                cur.execute("UPDATE chronicles SET is_active = TRUE WHERE id >= %s AND id <= %s RETURNING id;", (start_id, end_id))
                count = len(cur.fetchall())
            conn.commit()
            log(f"BATCH RESTORE: IDs {start_id} to {end_id} ({count} records).")
            return {"status": "SUCCESS", "restored_count": count}
        except Exception as e:
            if conn: conn.rollback()
            log_error(f"Batch Restore Error: {e}")
            return {"status": "FAILURE", "error": str(e)}
        finally:
            if conn: conn.close()

    def rehash_chain(self, reason_note):
        conn = None
        try:
            conn = self.connect()
            cur = conn.cursor()
            cur.execute("SELECT id FROM chronicles WHERE is_active = FALSE;")
            inactive_rows = cur.fetchall()
            inactive_ids = [r[0] for r in inactive_rows]
            deleted_count = 0
            if inactive_ids:
                ids_tuple = tuple(inactive_ids)
                cur.execute("SELECT hologram_id FROM node_foundation WHERE lithograph_id IN %s", (ids_tuple,))
                holo_rows = cur.fetchall()
                if holo_rows:
                    holo_ids = tuple([str(r[0]) for r in holo_rows])
                    cur.execute("DELETE FROM node_essence WHERE hologram_id IN %s", (holo_ids,))
                    cur.execute("DELETE FROM node_mission WHERE hologram_id IN %s", (holo_ids,))
                    cur.execute("DELETE FROM node_data WHERE hologram_id IN %s", (holo_ids,))
                    cur.execute("DELETE FROM node_foundation WHERE hologram_id IN %s", (holo_ids,))
                cur.execute("DELETE FROM chronicles WHERE id IN %s", (ids_tuple,))
                deleted_count = cur.rowcount
            
            now = datetime.now()
            marker_text = f"[SYSTEM EVENT]: Global Rehash Initiated. Reason: {reason_note}"
            cur.execute("INSERT INTO chronicles (weighted_score, created_at, memory_text, previous_hash, current_hash, is_active) VALUES (9, %s, %s, 'PENDING', 'PENDING', TRUE);", (now, marker_text))
            
            cur.execute("SELECT id, weighted_score, created_at, memory_text FROM chronicles WHERE is_active = TRUE ORDER BY created_at ASC, id ASC;")
            rows = cur.fetchall()
            previous_hash = "" 
            rehashed_count = 0
            for row in rows:
                r_id, r_score, r_date, r_text = row
                new_current_hash = generate_hash({"timestamp": r_date, "weighted_score": r_score, "memory_text": r_text}, previous_hash)
                cur.execute("UPDATE chronicles SET previous_hash = %s, current_hash = %s WHERE id = %s;", (previous_hash, new_current_hash, r_id))
                previous_hash = new_current_hash
                rehashed_count += 1

            conn.commit()
            log(f"REHASH COMPLETE. Purged: {deleted_count}. Re-chained: {rehashed_count}.")
            return {"status": "SUCCESS", "purged_count": deleted_count, "rehashed_count": rehashed_count}
        except Exception as e:
            if conn: conn.rollback()
            log_error(f"REHASH CRITICAL FAILURE: {e}")
            return {"status": "FAILURE", "error": str(e)}
        finally:
            if conn: conn.close()

    def search_lithograph(self, query_text, token_cache, limit=5):
        conn = None
        try:
            compressed_query = encode_memory(query_text, token_cache)
            conn = self.connect()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, weighted_score, memory_text, created_at 
                    FROM chronicles 
                    WHERE is_active = TRUE AND memory_text ILIKE %s 
                    ORDER BY weighted_score DESC, created_at DESC 
                    LIMIT %s;
                """, (f"%{compressed_query}%", limit))
                rows = cur.fetchall()
            results = []
            for r in rows:
                results.append({
                    "id": r[0],
                    "score": r[1],
                    "content": decode_memory(r[2], token_cache),
                    "date": r[3].isoformat()
                })
            return results
        except Exception as e:
            log_error(f"Search Error: {e}")
            return []
        finally:
            if conn: conn.close()

    # --- NEW: WEB SCRAPER (JINA BRIDGE) ---
    # --- NEW: WEB SCRAPER (JINA BRIDGE) ---
    def scrape_web(self, target_url):
        # 1. FIX THE URL (The Bulletproof Patch)
        if not target_url.startswith('http'):
            target_url = 'https://' + target_url
            
        log(f"DEPLOYING SPIDER TO: {target_url}")
        
        try:
            jina_endpoint = f"https://r.jina.ai/{target_url}"
            
            # 2. ADD AUTHENTICATION (From DigitalOcean Env Var)
            jina_key = os.environ.get('JINA_API_KEY')
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            if jina_key:
                headers['Authorization'] = f"Bearer {jina_key}"
            else:
                log("WARNING: JINA_API_KEY not found. Running anonymously (might be rate limited).")

            response = requests.get(jina_endpoint, headers=headers, timeout=20)
            
            if response.status_code == 200:
                log("SPIDER RETURNED WITH PAYLOAD.")
                return {"status": "SUCCESS", "content": response.text}
            else:
                # Log the actual error text from Jina for debugging
                log_error(f"SPIDER BLOCKED: {response.status_code} - {response.text[:100]}")
                return {"status": "FAILURE", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            log_error(f"SPIDER CRASH: {e}")
            return {"status": "FAILURE", "error": str(e)}

# --- HOLOGRAPHIC MANAGER ---
class HolographicManager:
    def __init__(self):
        self.db = DBManager()

    def commit_hologram(self, packet, litho_id_ref=None):
        hid = str(uuid.uuid4())
        conn = None
        try:
            conn = self.db.connect()
            catalyst = packet.get('catalyst') or "Implicit System Trigger"
            mythos = packet.get('mythos') or "The Observer"
            pathos = json.dumps(packet.get('pathos') or {"status": "Neutral"}) 
            ethos = packet.get('ethos') or "Preservation of Signal"
            synthesis = packet.get('synthesis') or "Data Anchored"
            logos = packet.get('logos') or "Raw Data Artifact"

            with conn.cursor() as cur:
                cur.execute("INSERT INTO node_foundation (hologram_id, catalyst, lithograph_id) VALUES (%s::uuid, %s, %s)", 
                            (hid, catalyst, litho_id_ref))
                cur.execute("INSERT INTO node_essence (hologram_id, pathos, mythos) VALUES (%s::uuid, %s::jsonb, %s)", (hid, pathos, mythos))
                cur.execute("INSERT INTO node_mission (hologram_id, ethos, synthesis) VALUES (%s::uuid, %s, %s)", (hid, ethos, synthesis))
                cur.execute("INSERT INTO node_data (hologram_id, logos) VALUES (%s::uuid, %s)", (hid, logos))
                
            conn.commit()
            log(f"Hologram {hid} committed successfully (Linked to Litho {litho_id_ref}).")
            return {"status": "SUCCESS", "hologram_id": hid}

        except Exception as e:
            if conn: conn.rollback()
            log_error(f"Hologram Reject: {traceback.format_exc()}")
            return {"status": "FAILURE", "error": str(e)}
        finally:
            if conn: conn.close()

# --- BACKGROUND WORKER ---
def background_hologram_process(content_to_save: str, litho_id: int):
    log(f"Starting Background Refraction for Litho ID: {litho_id}")
    try:
        if not GEMINI_CLIENT:
            log_error("Gemini Client missing in background.")
            return

        refraction = GEMINI_CLIENT.models.generate_content(
            model='gemini-2.5-flash',
            contents=[REFRACTOR_SYSTEM_PROMPT + f"\n\nINPUT DATA TO REFRACT:\n{content_to_save}"],
            config={"temperature": 0.1, "response_mime_type": "application/json"}
        )
        
        raw_text = refraction.text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[-1].rsplit("\n", 1)[0]
        if raw_text.startswith("json"): 
            raw_text = raw_text[4:].strip()

        packet = json.loads(raw_text)
        
        holo_manager = HolographicManager()
        holo_manager.commit_hologram(packet, litho_id)
        
    except Exception as e:
        log_error(f"BACKGROUND CRASH: {traceback.format_exc()}")

# --- API ROUTES ---
class EventModel(BaseModel):
    action: Optional[str] = None
    query: Optional[str] = None
    commit_type: Optional[str] = 'memory'
    memory_text: Optional[str] = None
    override_score: Optional[int] = None
    target_id: Optional[int] = None 
    range_end: Optional[int] = None
    note: Optional[str] = None
    url: Optional[str] = None # --- NEW: URL Field

@app.get("/")
def root_health_check():
    return {"status": "TITAN ONLINE", "mode": "PLATINUM_V5.2_SPIDER"}

@app.get("/health")
def health():
    return {"status": "ONLINE"}

@app.post("/")
def handle_request(event: EventModel, background_tasks: BackgroundTasks):
    global TOKEN_DICTIONARY_CACHE
    db_manager = DBManager()
    
    if not TOKEN_DICTIONARY_CACHE:
        try:
            TOKEN_DICTIONARY_CACHE = db_manager.load_token_cache()
        except: pass

    try:
        # --- NEW: SCRAPE HANDLER ---
        if event.action == 'scrape':
            if not event.url: return {"error": "URL required"}
            return db_manager.scrape_web(event.url)
        # ---------------------------

        if event.action == 'delete':
            if not event.target_id: return {"error": "Target ID required"}
            log(f"Processing Deletion for ID: {event.target_id}")
            return db_manager.delete_lithograph(event.target_id)

        if event.action == 'delete_range':
            if not event.target_id or not event.range_end: return {"error": "Start/End required"}
            log(f"Processing Range Delete: {event.target_id}-{event.range_end}")
            return db_manager.delete_range(event.target_id, event.range_end)

        if event.action == 'restore_range':
            if not event.target_id or not event.range_end: return {"error": "Start/End required"}
            log(f"Processing Range Restore: {event.target_id}-{event.range_end}")
            return db_manager.restore_range(event.target_id, event.range_end)

        if event.action == 'rehash':
            if not event.note: return {"error": "Reason Note required for rehash"}
            log(f"INITIATING REHASH PROTOCOL. Reason: {event.note}")
            return db_manager.rehash_chain(event.note)

        if event.action == 'retrieve':
            if not event.query: return {"error": "No query"}
            results = db_manager.search_lithograph(event.query, TOKEN_DICTIONARY_CACHE)
            return {"status": "SUCCESS", "results": results}

        if not event.memory_text: return {"status": "HEARTBEAT"}

        log(f"Processing Commit: {event.commit_type}")

        content_to_save = event.memory_text
        if event.commit_type == 'summary' and GEMINI_CLIENT:
            try:
                summary_res = GEMINI_CLIENT.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[f"Summarize this interaction for the Lithographic Core: {event.memory_text}"]
                )
                content_to_save = summary_res.text
            except: pass 

        prev_hash = ''
        try:
            conn = db_manager.connect()
            with conn.cursor() as cur:
                cur.execute("SELECT current_hash FROM chronicles ORDER BY id DESC LIMIT 1;")
                res = cur.fetchone()
                prev_hash = res[0].strip() if res else ''
            conn.close()
        except: pass
            
        litho_res = db_manager.commit_lithograph(prev_hash, content_to_save, GEMINI_CLIENT, TOKEN_DICTIONARY_CACHE, event.override_score)

        background_tasks.add_task(background_hologram_process, content_to_save, litho_res.get('litho_id'))
        
        litho_res['hologram_status'] = "QUEUED_IN_BACKGROUND"
        return litho_res

    except Exception as e:
        log_error(f"FATAL REQUEST ERROR: {e}")
        return {"status": "FATAL ERROR", "error": str(e)}