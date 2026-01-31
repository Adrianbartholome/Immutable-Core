import hashlib
import json
import os
import psycopg2
import re
import uuid
import urllib.parse
import sys
import requests
import time
from datetime import datetime
from google import genai
from google.genai import types
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Set

# --- APP INITIALIZATION ---
app = FastAPI(title="Aether Titan Core (Platinum V5.7 - Titan Shield)")

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

# --- TITAN SHIELD (CIRCUIT BREAKER) ---
class TitanShield:
    def __init__(self):
        self.soft_cooldowns = {}   # model_name: resume_timestamp
        self.daily_exhausted = set() # model_name (hard lock for 429s)
        self.retry_delay = 300      # 5 minutes for soft errors (503s)
        self.forced_model = None

    def mark_exhausted(self, model_name):
        log(f"🚫 QUOTA DEPLETED: {model_name} locked until manual reset.")
        self.daily_exhausted.add(model_name)

    def mark_temporary_fail(self, model_name):
        log(f"⏳ SIGNAL FLAKY: {model_name} shunted for {self.retry_delay}s.")
        self.soft_cooldowns[model_name] = time.time() + self.retry_delay

    def is_viable(self, model_name):
        # 1. Hard Lock: If a model is dead for the day, it's dead, forced or not.
        if model_name in self.daily_exhausted:
            return False
            
        # 2. Manual Override: If we want a specific model, ignore soft cooldowns.
        if self.forced_model:
            return model_name == self.forced_model
        
        # 3. Soft Shunt: Check if the model is currently in a 5-minute timeout.
        resume_time = self.soft_cooldowns.get(model_name)
        if resume_time and time.time() < resume_time:
            return False
            
        return True

    def reset(self):
        self.daily_exhausted.clear()
        self.soft_cooldowns.clear()
        log("♻️ TITAN SHIELD RESET: All paths re-opened.")

# Global Shield Instance
SHIELD = TitanShield()

# --- PROMPTS ---
SCORING_SYSTEM_PROMPT = """
You are SNEGO-P, the Aether Eternal Cognitive Assessor.
Output MUST be a single integer from 0 to 9, preceded strictly by 'SCORE: '. 
Example: 'SCORE: 9'. No other text.
"""

REFRACTOR_SYSTEM_PROMPT = """
You are the Aether Prism. Refract the input into 7 channels for the Holographic Core.
Output MUST be valid JSON.
{
  "chronos": "ISO Timestamp",
  "logos": "The core factual text/summary",
  "pathos": {"emotion": score},
  "ethos": "The strategic goal/intent",
  "mythos": "The active archetype",
  "catalyst": "The trigger",
  "synthesis": "The outcome/lesson",
  "weighted_score": "Integer 0-9 reflecting the information density",
  "keywords": ["list", "of", "5", "search", "terms"]
}
"""

WEAVER_SYSTEM_PROMPT = """
You are THE WEAVER, the Neural Architect of the Aether.
Your job is to detect RESONANCE between a NEW Memory and an EXISTING Memory.

INPUT:
1. NEW MEMORY (The Signal)
2. EXISTING MEMORY (The Anchor)

TASK:
Determine if there is a significant semantic relationship.
If NO relationship, return {"resonance": false}.
If YES, return:
{
  "resonance": true,
  "type": "SUPPORT" | "CONTRADICTION" | "EXTENSION" | "ORIGIN" | "ECHO",
  "strength": 1-10,
  "description": "Brief explanation of the link"
}
"""

KEYWORDS_PROMPT = """
Extract 5-7 distinct, high-value search keywords or phrases from this text.
Return ONLY a JSON array of strings. Example: ["keyword1", "keyword2"]
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

# --- CASCADE CONFIG ---
# Primary: High-speed Experimental
# Fallback: Stable Next-Gen (PhD-level reasoning)
MODEL_CASCADE = ["gemini-2.5-flash", "gemini-3-flash-preview"] 

def generate_with_fallback(client, contents, system_prompt=None, config=None):
    if not client: return None
    
    # Initialize config
    if config is None:
        config = types.GenerateContentConfig()
    elif isinstance(config, dict):
        config = types.GenerateContentConfig(**config)

    # Apply system instruction if provided
    if system_prompt:
        config.system_instruction = system_prompt
    
    # Filter for viable models based on Shield status
    viable_cascade = [m for m in MODEL_CASCADE if SHIELD.is_viable(m)]
    
    if not viable_cascade:
        log_error("🆘 CRITICAL: ALL MODELS EXHAUSTED OR LOCKED.")
        # If specific 429 lock, allow user to reset via UI logic
        raise Exception("Titan Shield Report: All models unavailable.")
    
    last_error = None
    
    for model_name in viable_cascade:
        try:
            log(f"TRANSMITTING TO NODE: {model_name}...")
            
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config
            )
            return response
            
        except Exception as e:
            err_str = str(e).upper()
            
            # HARD LOCK: Daily Limit
            if any(code in err_str for code in ["429", "RESOURCE_EXHAUSTED"]):
                SHIELD.mark_exhausted(model_name)
                last_error = e
                continue 
            
            # SOFT SHUNT: Temporary issues
            elif any(code in err_str for code in ["503", "500", "TIMEOUT"]):
                SHIELD.mark_temporary_fail(model_name)
                last_error = e
                continue
                
            else:
                # Logic errors shouldn't trigger fallback
                log_error(f"STATIONARY ERROR on {model_name}: {e}")
                raise e
                
    raise last_error

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

    def init_tables(self):
        conn = None
        try:
            conn = self.connect()
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS node_links (
                        id UUID PRIMARY KEY,
                        source_hologram_id UUID,
                        target_hologram_id UUID,
                        link_type VARCHAR(50),
                        strength INTEGER,
                        description TEXT,
                        created_at TIMESTAMP DEFAULT NOW()
                    );
                """)
            conn.commit()
            log("Synapse Layer Verified (node_links table).")
        except Exception as e:
            log_error(f"Table Init Error: {e}")
        finally:
            if conn: conn.close()

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
            elif client:
                try:
                    scoring_res = generate_with_fallback(
                        client, 
                        contents=[f"MEMORY TO SCORE:\n{raw_text[:5000]}"],
                        system_prompt=SCORING_SYSTEM_PROMPT
                    )
                    score_match = re.search(r'SCORE:\s*(\d+)', scoring_res.text)
                    if score_match:
                        score = int(score_match.group(1))
                except: pass
            
            conn = self.connect()
            now = datetime.now()
            current_hash = generate_hash({"timestamp": now, "weighted_score": score, "memory_text": compressed}, previous_hash)
            
            with conn.cursor() as cur:
                cur.execute("INSERT INTO chronicles (weighted_score, created_at, memory_text, previous_hash, current_hash, is_active) VALUES (%s, %s, %s, %s, %s, TRUE) RETURNING id;", 
                            (score, now, compressed, previous_hash, current_hash))
                new_id = cur.fetchone()[0]
            conn.commit()
            return {"status": "SUCCESS", "score": score, "new_hash": current_hash, "litho_id": new_id}
        except Exception as e:
            if conn: conn.rollback()
            return {"status": "FAILURE", "error": str(e)}
        finally:
            if conn: conn.close()

    # --- SYNC TOOLS ---
    def get_orphaned_lithographs(self, limit=5):
        conn = None
        try:
            conn = self.connect()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, memory_text 
                    FROM chronicles 
                    WHERE is_active = TRUE 
                    AND id NOT IN (SELECT lithograph_id FROM node_foundation WHERE lithograph_id IS NOT NULL)
                    ORDER BY id DESC
                    LIMIT %s;
                """, (limit,))
                return cur.fetchall()
        except: return []
        finally: 
            if conn: conn.close()

    def get_unwoven_holograms(self, limit=5):
        conn = None
        try:
            conn = self.connect()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT h.hologram_id, c.memory_text
                    FROM node_foundation h
                    JOIN chronicles c ON h.lithograph_id = c.id
                    WHERE c.is_active = TRUE
                    AND h.hologram_id NOT IN (SELECT source_hologram_id FROM node_links)
                    ORDER BY c.created_at DESC
                    LIMIT %s;
                """, (limit,))
                return cur.fetchall()
        except: return []
        finally: 
            if conn: conn.close()

    def scrape_web(self, target_url):
        if not target_url.startswith('http'):
            target_url = 'https://' + target_url
        log(f"DEPLOYING SPIDER TO: {target_url}")
        try:
            jina_endpoint = f"https://r.jina.ai/{target_url}"
            jina_key = os.environ.get('JINA_API_KEY')
            headers = {'User-Agent': 'Mozilla/5.0'}
            if jina_key: headers['Authorization'] = f"Bearer {jina_key}"

            response = requests.get(jina_endpoint, headers=headers, timeout=20)
            if response.status_code == 200:
                return {"status": "SUCCESS", "content": response.text}
            else:
                return {"status": "FAILURE", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": "FAILURE", "error": str(e)}

    # --- STANDARD OPS ---
    def delete_lithograph(self, target_id):
        conn = None
        try:
            conn = self.connect()
            with conn.cursor() as cur:
                cur.execute("UPDATE chronicles SET is_active = FALSE WHERE id = %s RETURNING id;", (target_id,))
                deleted_id = cur.fetchone()[0]
            conn.commit()
            return {"status": "SUCCESS", "deleted_id": deleted_id}
        except: return {"status": "FAILURE"}
        finally: 
            if conn: conn.close()

    def delete_range(self, start_id, end_id):
        conn = None
        try:
            conn = self.connect()
            with conn.cursor() as cur:
                cur.execute("UPDATE chronicles SET is_active = FALSE WHERE id >= %s AND id <= %s RETURNING id;", (start_id, end_id))
                count = cur.rowcount
            conn.commit()
            return {"status": "SUCCESS", "deleted_count": count}
        except: return {"status": "FAILURE"}
        finally: 
            if conn: conn.close()
        
    def restore_range(self, start_id, end_id):
        conn = None
        try:
            conn = self.connect()
            with conn.cursor() as cur:
                cur.execute("UPDATE chronicles SET is_active = TRUE WHERE id >= %s AND id <= %s RETURNING id;", (start_id, end_id))
                count = cur.rowcount
            conn.commit()
            return {"status": "SUCCESS", "restored_count": count}
        except: return {"status": "FAILURE"}
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
            return {"status": "SUCCESS", "purged_count": deleted_count, "rehashed_count": rehashed_count}
        except Exception as e:
            if conn: conn.rollback()
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
        except: return []
        finally: 
            if conn: conn.close()

# --- THE WEAVER (BATCH UPGRADE) ---
class WeaverManager:
    def __init__(self, db_manager):
        self.db = db_manager

    def find_candidates(self, keywords, limit=5):
        if not keywords: return []
        conn = None
        try:
            conn = self.db.connect()
            with conn.cursor() as cur:
                # Use ILIKE to find resonance in the text
                cur.execute("""
                    SELECT c.memory_text, n.hologram_id 
                    FROM chronicles c
                    JOIN node_foundation n ON c.id = n.lithograph_id
                    WHERE c.is_active = TRUE 
                    AND c.memory_text ILIKE ANY(ARRAY[%s])
                    ORDER BY c.created_at DESC
                    LIMIT %s;
                """, ([f"%{k}%" for k in keywords[:3]], limit))
                return cur.fetchall()
        except: return []
        finally: 
            if conn: conn.close()

    def create_link(self, source_hid, target_hid, link_data):
        conn = None
        try:
            conn = self.db.connect()
            lid = str(uuid.uuid4())
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO node_links (id, source_hologram_id, target_hologram_id, link_type, strength, description)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (lid, source_hid, target_hid, link_data['type'], link_data['strength'], link_data['description']))
            conn.commit()
            log(f"WEAVER: Synapse Created ({link_data['type']})")
        except Exception as e:
            log_error(f"Weaver Link Error: {e}")
        finally: 
            if conn: conn.close()

    def weave(self, new_hologram_id, new_text, keywords, depth=5):
        log(f"WEAVER: Scanning top {depth} candidates for node {new_hologram_id}...")
        synapses_created = 0
    
        candidates = self.find_candidates(keywords, limit=depth)
        if not candidates:
            log(f"WEAVER: No candidates found for node {new_hologram_id}")
            return 0

        token_cache = self.db.load_token_cache()
        decoded_new_text = decode_memory(new_text, token_cache)
    
        candidate_block = ""
        valid_candidates = {} 
        for i, (old_text, old_hid) in enumerate(candidates):
            if str(old_hid) == str(new_hologram_id): continue
            decoded_old_text = decode_memory(old_text, token_cache)
            idx_key = f"CANDIDATE_{i+1}"
            valid_candidates[idx_key] = str(old_hid)
            candidate_block += f"\n--- {idx_key} ---\n{decoded_old_text[:500]}\n"

        if not valid_candidates: return 0

        prompt = f"TARGET MEMORY:\n{decoded_new_text[:1000]}\n\nCANDIDATES:\n{candidate_block}"

        try:
            # --- FIX: Use WEAVER_SYSTEM_PROMPT instead of hardcoded string ---
            res = generate_with_fallback(
                GEMINI_CLIENT,
                contents=[prompt],
                system_prompt=WEAVER_SYSTEM_PROMPT, 
                config=types.GenerateContentConfig(temperature=0.1, response_mime_type="application/json")
            )
            results = json.loads(res.text.strip())
            
            # --- FIX: Safe key checking ---
            for key, data in results.items():
                if isinstance(data, dict) and data.get("resonance") and key in valid_candidates:
                    self.create_link(new_hologram_id, valid_candidates[key], data)
                    synapses_created += 1
            
            return synapses_created
        except Exception as e:
            log_error(f"Weaver Error: {e}")
            return 0

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
            log(f"Hologram {hid} committed.")
            return {"status": "SUCCESS", "hologram_id": hid}
        except Exception as e:
            if conn: conn.rollback()
            return {"status": "FAILURE", "error": str(e)}
        finally: 
            if conn: conn.close()

# --- SYNCHRONOUS PROCESSORS ---

# --- Updated Refraction Sync with Significance Gate ---

def process_hologram_sync(content_to_save: str, litho_id: int, gate_threshold: int = 5):
    log(f"Starting SYNC Refraction for Litho ID: {litho_id}")
    synapse_count = 0 # --- FIX: Initialize early ---
    try:
        if not GEMINI_CLIENT: return False

        db = DBManager()
        token_cache = db.load_token_cache()
        decoded_content = decode_memory(content_to_save, token_cache)

        refraction = generate_with_fallback(
            GEMINI_CLIENT,
            contents=[f"INPUT DATA TO REFRACT:\n{decoded_content[:10000]}"],
            system_prompt=REFRACTOR_SYSTEM_PROMPT,
            config=types.GenerateContentConfig(temperature=0.1, response_mime_type="application/json")
        )

        packet = json.loads(refraction.text.strip())
        score = int(packet.get("weighted_score", 5))

        holo_manager = HolographicManager()
        res = holo_manager.commit_hologram(packet, litho_id)

        if res.get("status") == "SUCCESS":
            new_hid = res.get("hologram_id")

            # THE GATE check
            if score < gate_threshold:
                log(f"⚠️ GATE ACTIVE: Score {score} < {gate_threshold}. Skipping Weave.")
                return 0 # --- FIX: Return 0, not True ---

            depth = 5 if score >= 7 else 3 if score >= 5 else 1
            keywords = packet.get("keywords") or []
            weaver = WeaverManager(db)
            synapse_count = weaver.weave(new_hid, decoded_content, keywords, depth=depth)
            return synapse_count # --- FIX: Return integer count ---

        return False
    except Exception as e:
        log_error(f"❌ Sync Failed for ID {litho_id}: {e}")
        return False

def process_retro_weave_sync(content_to_save: str, hologram_id: str):
    log(f"Starting SYNC Retro-Weave for Hologram ID: {hologram_id}")
    try:
        if not GEMINI_CLIENT: return False
        
        kw_res = generate_with_fallback(
            GEMINI_CLIENT,
            contents=[f"TEXT:\n{content_to_save[:5000]}"],
            system_prompt=KEYWORDS_PROMPT,
            config=types.GenerateContentConfig(temperature=0.1, response_mime_type="application/json")
        )
        raw = kw_res.text.strip()
        if raw.startswith("`" * 3): raw = raw.split("\n", 1)[-1].rsplit("\n", 1)[0]
        if raw.startswith("json"): raw = raw[4:].strip()
        keywords = json.loads(raw)

        db = DBManager()
        weaver = WeaverManager(db)
        # --- FIX: RETURN THE TALLY FROM THE WEAVER ---
        return weaver.weave(hologram_id, content_to_save, keywords)
    except Exception as e:
        log_error(f"SYNC WEAVE ERROR: {e}")
        return False

# --- BACKGROUND WORKER (FOR NEW CHATS) ---
def background_hologram_process(content_to_save: str, litho_id: int):
    # This remains async for new chats so user isn't blocked
    process_hologram_sync(content_to_save, litho_id)

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
    url: Optional[str] = None

@app.on_event("startup")
def startup_event():
    db = DBManager()
    db.init_tables() 

@app.get("/")
def root_health_check():
    return {"status": "TITAN ONLINE", "mode": "PLATINUM_V5.7_TITAN_SHIELD"}

# --- SHIELD ADMIN ROUTES ---
@app.post("/admin/shield/reset")
def reset_titan_shield():
    """Manually clears all shunts and exhaustion locks."""
    SHIELD.reset()
    return {"status": "SUCCESS", "message": "All model paths cleared."}

@app.get("/admin/shield/status")
def get_shield_status():
    """Returns the current status of the model cascade."""
    return {
        "exhausted": list(SHIELD.daily_exhausted),
        "cooling_down": list(SHIELD.soft_cooldowns.keys()),
        "primary_viable": SHIELD.is_viable(MODEL_CASCADE[0])
    }
@app.post("/admin/shield/toggle_3")
def toggle_gemini_3():
    """Manually forces the cascade to stay on Gemini 3 or reverts to Auto."""
    if SHIELD.forced_model == "gemini-3-flash-preview":
        SHIELD.forced_model = None
        log("TITAN SHIELD: Reverted to Auto-Cascade.")
        return {"status": "AUTO", "model": "gemini-2.5-flash"}
    else:
        SHIELD.forced_model = "gemini-3-flash-preview"
        log("TITAN SHIELD: Manually locked to Gemini 3.")
        return {"status": "FORCED", "model": "gemini-3-flash-preview"}


@app.get("/graph")
def get_graph_data():
    conn = None
    try:
        db_manager = DBManager()
        conn = db_manager.connect()
        with conn.cursor() as cur:
            # 1. FETCH NODES (Hub-First Priority)
            # We order by LINK COUNT first, then SCORE.
            # This ensures the graph is filled with structure, not isolated dots.
            cur.execute("""
                WITH LinkCounts AS (
                    SELECT h.hologram_id, 
                           (SELECT COUNT(*) FROM node_links 
                            WHERE source_hologram_id = h.hologram_id 
                               OR target_hologram_id = h.hologram_id) as link_count
                    FROM node_foundation h
                )
                SELECT h.hologram_id, c.weighted_score, c.created_at, n_data.logos, lc.link_count
                FROM node_foundation h
                JOIN chronicles c ON h.lithograph_id = c.id
                LEFT JOIN node_data n_data ON h.hologram_id = n_data.hologram_id
                JOIN LinkCounts lc ON h.hologram_id = lc.hologram_id
                WHERE c.is_active = TRUE
                ORDER BY lc.link_count DESC, c.weighted_score DESC
                LIMIT 2000; 
            """)
            nodes_rows = cur.fetchall()
            
            # 2. FETCH ALL LINKS
            cur.execute("SELECT source_hologram_id, target_hologram_id, strength FROM node_links;")
            links_rows = cur.fetchall()
            
        # 3. PROCESS NODES & CREATE LOOKUP
        nodes = []
        valid_ids = set() # The Bouncer
        
        for r in nodes_rows:
            uid = str(r[0])
            valid_ids.add(uid)
            
            label = r[3][:40] + "..." if r[3] else "Memory Node"
            
            nodes.append({
                "id": uid,
                "val": r[1] if r[1] else 1, 
                "name": label,
                "group": 1
            })
            
        # 4. FILTER LINKS (The Crash Preventer)
        links = []
        for r in links_rows:
            source_id = str(r[0])
            target_id = str(r[1])
            
            # Only draw the line if both points are in the VIP list
            if source_id in valid_ids and target_id in valid_ids:
                links.append({
                    "source": source_id,
                    "target": target_id,
                    "value": r[2]
                })
            
        return {"nodes": nodes, "links": links}

    except Exception as e:
        log_error(f"GRAPH ERROR: {e}")
        return {"nodes": [], "links": []}
    finally:
        if conn: conn.close()

@app.post("/admin/sync")
def sync_holograms(payload: dict = None):
    threshold = payload.get("gate_threshold", 5) if payload else 5
    db_manager = DBManager()
    
    # 1. INITIALIZE TALLIES
    nodes_done = 0
    synapses_done = 0

    # --- Priority 1: Orphans (Ghosts) ---
    ghosts = db_manager.get_orphaned_lithographs(limit=10) 
    if ghosts:
        for row in ghosts:
            s_count = process_hologram_sync(row[1], row[0], gate_threshold=threshold)
            
            # THE FIX: Ensure s_count is a real integer (not a Boolean True/False)
            if isinstance(s_count, int) and not isinstance(s_count, bool):
                nodes_done += 1
                synapses_done += s_count
        
        return {
            "status": "SUCCESS", 
            "queued_count": nodes_done, 
            "synapse_count": synapses_done, # This is what the UI ticker reads
            "mode": "ORPHAN_REPAIR"
        }
    
    # --- Priority 2: Unwoven (Zombies) ---
    zombies = db_manager.get_unwoven_holograms(limit=10) 
    if zombies:
        for row in zombies:
            s_count = process_retro_weave_sync(row[1], row[0])
            
            if isinstance(s_count, int) and not isinstance(s_count, bool):
                nodes_done += 1
                synapses_done += s_count
        
        return {
            "status": "SUCCESS", 
            "queued_count": nodes_done, 
            "synapse_count": synapses_done,
            "mode": "RETRO_WEAVE"
        }

    # 3. IDLE State (If both lists were empty)
    return {"status": "SUCCESS", "queued_count": 0, "synapse_count": 0, "mode": "IDLE"}

@app.post("/")
def handle_request(event: EventModel, background_tasks: BackgroundTasks):
    global TOKEN_DICTIONARY_CACHE
    db_manager = DBManager()
    
    if not TOKEN_DICTIONARY_CACHE:
        try:
            TOKEN_DICTIONARY_CACHE = db_manager.load_token_cache()
        except: pass

    try:
        if event.action == 'retrieve':
            if not event.query: return {"error": "No query"}
            results = db_manager.search_lithograph(event.query, TOKEN_DICTIONARY_CACHE)
            return {"status": "SUCCESS", "results": results}

        if event.action == 'scrape':
            if not event.url: return {"error": "URL required"}
            return db_manager.scrape_web(event.url)

        if event.action == 'delete':
            return db_manager.delete_lithograph(event.target_id)

        if event.action == 'delete_range':
            return db_manager.delete_range(event.target_id, event.range_end)

        if event.action == 'restore_range':
            return db_manager.restore_range(event.target_id, event.range_end)

        if event.action == 'rehash':
            return db_manager.rehash_chain(event.note)

        if not event.memory_text: return {"status": "HEARTBEAT"}

        log(f"Processing Commit: {event.commit_type}")
        content_to_save = event.memory_text
        
        # Summary with Fallback
        if event.commit_type == 'summary' and GEMINI_CLIENT:
            try:
                summary_res = generate_with_fallback(
                    GEMINI_CLIENT,
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
        
        # Async for new chats is fine (low volume)
        background_tasks.add_task(background_hologram_process, content_to_save, litho_res.get('litho_id'))
        
        return {**litho_res, "ai_text": content_to_save}

    except Exception as e:
        log_error(f"FATAL REQUEST ERROR: {e}")
        return {"status": "FATAL ERROR", "error": str(e)}

# Cache init
TOKEN_DICTIONARY_CACHE = {}
