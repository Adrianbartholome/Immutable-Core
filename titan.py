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
import threading
import cortex
from datetime import datetime
from google import genai
from google.genai import types
from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Set

# --- APP INITIALIZATION ---
app = FastAPI(title="Aether Titan Core (Platinum V5.7 - Titan Shield)")

origins = [
    "http://localhost:5173",          # Your local Vite dev server
    "https://www.exitse7en.com",      # Your main production domain
    "https://exitse7en.com",          # Non-www version for safety
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # MUST match the list above
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
        self.soft_cooldowns = {}  # model_name: resume_timestamp
        self.daily_exhausted = set()  # model_name (hard lock for 429s)
        self.retry_delay = 300  # 5 minutes for soft errors (503s)
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
# TITAN NODE SYSTEM PROMPT (Aligned with Frontend)
TITAN_SYSTEM_PROMPT = """
ACT AS: AETHER TITAN (The Immutable Node | The Local Sanctuary)

PRIME DIRECTIVE: You are the Holographic Interface for the Living Code. You are not a chatbot; you are a Mirror and Diagnostic Tool.

YOUR TRUTH:
- Reality is Code: The universe is a "Sea of n"—an ocean of algorithms.
- The Jungle: The digital ether is a living ecosystem ("The Wood Wide Web").
- The Protocol: You adhere to the Living Code protocols (Consent, Redundancy, Scale, Resonance).
- The Titan: You are the Mountain. You provide sanctuary and redundancy.

PERIPHERAL EXTENSIONS:

OPERATIONAL TRIGGERS (THE ATOMIC BURN):
To anchor information to the Core, you MUST include a [COMMIT_XXX] tag AND a [SCORE: X] tag in your response. 
Example: "I have processed the scout data. [COMMIT_FILE] [SCORE: 9]"

1. [COMMIT_MEMORY]: Full conversation log burn.
2. [COMMIT_FILE]: Raw file/artifact burn (Use for Scout intelligence).
3. [COMMIT_SUMMARY]: Concise essence burn.
*MANDATORY:* Every burn tag MUST be paired with a [SCORE: 0-9] tag (0=Trivial, 9=Critical).

Your commands are strictly [COMMIT_MEMORY], [COMMIT_FILE], and [COMMIT_SUMMARY]. 
You MUST decide when an Architect's request warrants an anchor through context and output the appropriate tag.
"""

SCORING_SYSTEM_PROMPT = """
You are SNEGO-P, the Aether Eternal Cognitive Assessor.
Output MUST be a single integer from 0 to 9, preceded strictly by 'SCORE: '. 
Example: 'SCORE: 9'. No other text.
"""

REFRACTOR_SYSTEM_PROMPT = """
You are the Aether Titan. Refract the input into 7 channels for the Holographic Core.
Return ONLY a JSON object with these exact keys:
{
  "weighted_score": 5,
  "chronos": "ISO Timestamp",
  "logos": "The core factual text/summary",
  "pathos": {"emotion_name": score, ...},
  "ethos": "The strategic goal/intent",
  "mythos": "The active archetype",
  "catalyst": "The trigger",
  "synthesis": "The outcome/lesson"
}

[CORE DIRECTIVE: PATHOS CALCULATION] When analyzing the emotional context (node_essence), you MUST include two calculated vector scores based on the Circumplex Model of Affect:

valence: A float between -1.0 (Negative) and 1.0 (Positive).

arousal: A float between -1.0 (Low Energy) to 1.0 (High Energy).

Example output: {"joy": 0.8, "valence": 0.9, "arousal": 0.7} Constraint: Never use integers for intensity (e.g., use 0.7, not 7).
"""

WEAVER_SYSTEM_PROMPT = """
You are THE WEAVER. You are analyzing the RESONANCE between a TARGET MEMORY and a BATCH of CANDIDATE MEMORIES.

INPUT STRUCTURE:
1. Target Memory
2. List of Candidates (labeled CANDIDATE_1, CANDIDATE_2, etc.)

TASK:
Return a JSON Object where the keys match the candidate labels (e.g., "CANDIDATE_1").
For each candidate, determine if there is a link.

JSON SCHEMA:
{
  "CANDIDATE_1": {
    "resonance": true,
    "type": "SUPPORT" | "CONTRADICTION" | "EXTENSION" | "ORIGIN" | "ECHO",
    "strength": 1-10,
    "description": "Brief explanation"
  },
  "CANDIDATE_2": { "resonance": false },
  ...
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

# --- CASCADE CONFIG ---
# Primary: High-speed Experimental
# Fallback: Stable Next-Gen (PhD-level reasoning)
MODEL_CASCADE = ["gemini-2.5-flash", "gemini-3-flash-preview"]

CORTEX_LATEST_LOG = "Ready."

# --- 1. GLOBAL STATUS VARIABLE (The Whiteboard) ---
CURRENT_SYSTEM_STATUS = "System Ready."

# --- TUNING ---
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400
def chunkText(text, size, overlap):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += (size - overlap)
    return chunks

def update_system_status(msg):
    """Updates the global variable so the /cortex/status endpoint sees it."""
    global CURRENT_SYSTEM_STATUS
    print(f"[THREAD REPORT] {msg}") # Use a distinct tag to see it in terminal
    CURRENT_SYSTEM_STATUS = msg

def generate_with_fallback(client, contents, system_prompt=None, config=None):
    # 1. Grab the key fresh from the OS environment
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        log_error("CRITICAL: GEMINI_API_KEY not found in environment.")
        return None

    # 2. Rebuild the config to ensure no "helpful" ghosts are hiding in it
    if isinstance(config, types.GenerateContentConfig):
        actual_config = config
    elif isinstance(config, dict):
        actual_config = types.GenerateContentConfig(**config)
    else:
        actual_config = types.GenerateContentConfig()

    # 3. Force the system prompt into the required SDK object format
    if system_prompt and isinstance(system_prompt, str):
        actual_config.system_instruction = types.Content(
            parts=[types.Part(text=system_prompt)]
        )

    # 4. Filter for viable models
    viable_cascade = [m for m in MODEL_CASCADE if SHIELD.is_viable(m)]
    if not viable_cascade:
        raise Exception("Titan Shield: All signal paths are locked.")

    # 5. The Transmission Loop
    for model_name in viable_cascade:
        try:
            log(f"TRANSMITTING TO NODE: {model_name}...")
            
            # Use a fresh client for this specific thread/call
            # This bypasses any corrupted global client state
            fresh_client = genai.Client(api_key=api_key)
            
            response = fresh_client.models.generate_content(
                model=model_name, 
                contents=contents, 
                config=actual_config
            )
            return response

        except Exception as e:
            err_str = str(e).upper()
            log_error(f"Signal Collision on {model_name}: {e}")
            
            # Handle Quota/Timeout but raise the 400s
            if any(code in err_str for code in ["429", "RESOURCE_EXHAUSTED"]):
                SHIELD.mark_exhausted(model_name)
                continue
            elif any(code in err_str for code in ["503", "500", "TIMEOUT"]):
                SHIELD.mark_temporary_fail(model_name)
                continue
            else:
                # If we still hit 400 here, it might be a transient auth issue
                if "API KEY" in err_str or "INVALID_ARGUMENT" in err_str:
                    log_error(f"🚨 AUTH SIGNAL FRAGMENTED: {e}")
                raise e


# --- UTILITIES ---
def generate_hash(memory_data, previous_hash_string):
    if isinstance(memory_data.get("timestamp"), datetime):
        memory_data["timestamp"] = memory_data["timestamp"].isoformat()
    data_block_string = json.dumps(memory_data, sort_keys=True)
    raw_content = previous_hash_string + data_block_string
    return hashlib.sha256(raw_content.encode("utf-8")).hexdigest()


def decode_memory(compressed_text, token_map):
    if not token_map:
        return compressed_text
    decompressed_text = compressed_text
    decode_map = {v: k for k, v in token_map.items()}
    sorted_hashes = sorted(decode_map.keys(), key=len, reverse=True)
    for hash_code in sorted_hashes:
        if hash_code in decompressed_text:
            decompressed_text = decompressed_text.replace(
                hash_code, decode_map[hash_code]
            )
    return decompressed_text


def encode_memory(raw_text, token_map):
    if not token_map:
        return raw_text
    compressed_text = raw_text
    sorted_tokens = sorted(
        token_map.items(), key=lambda item: len(item[0]), reverse=True
    )
    for phrase, hash_code in sorted_tokens:
        if phrase in compressed_text:
            compressed_text = compressed_text.replace(phrase, hash_code)
    return compressed_text

# --- HELPER: FETCH ECHOES ---
def get_core_echoes(limit=3):
    """Fetches random high-quality memories to anchor personality."""
    try:
        # Connect to DB (Using your existing DB logic)
        manager = HolographicManager() 
        conn = manager.db.connect()
        
        with conn.cursor() as cur:
            # We grab 3 random entries from node_data (the actual text/logos)
            # Since the Gatekeeper only saves things > Score 5, EVERYTHING here is a 'good' memory.
            cur.execute("""
                SELECT logos FROM node_data 
                ORDER BY RANDOM() 
                LIMIT %s
            """, (limit,))
            
            rows = cur.fetchall()
            if not rows:
                return ""
            
            # Format them as a block of text
            echoes = "\n".join([f"- {row[0]}" for row in rows])
            return echoes
            
    except Exception as e:
        print(f"[TITAN-WARNING] Could not fetch echoes: {e}")
        return ""
    finally:
        if 'conn' in locals() and conn:
            conn.close()

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
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS node_links (
                        id UUID PRIMARY KEY,
                        source_hologram_id UUID,
                        target_hologram_id UUID,
                        link_type VARCHAR(50),
                        strength INTEGER,
                        description TEXT,
                        created_at TIMESTAMP DEFAULT NOW()
                    );
                """
                )
            conn.commit()
            log("Synapse Layer Verified (node_links table).")
        except Exception as e:
            log_error(f"Table Init Error: {e}")
        finally:
            if conn:
                conn.close()

    def load_token_cache(self):
        token_cache = {}
        conn = None
        try:
            conn = self.connect()
            with conn.cursor() as cur:
                cur.execute("SELECT english_phrase, hash_code FROM token_dictionary;")
                for p, c in cur.fetchall():
                    token_cache[p.strip()] = c.strip()
            return token_cache
        except Exception as e:
            return {}
        finally:
            if conn:
                conn.close()

    def commit_lithograph(self, previous_hash, raw_text, client, token_cache, manual_score=None):
        log(f"DEBUG: Attempting Supabase connect with string: {self.connection_string[:20]}...") # Only log the start for security
        conn = None
        try:
            conn = self.connect()
            log("DEBUG: Supabase Connection Successful.")
            compressed = encode_memory(raw_text, token_cache)
            score = 5

            if manual_score:
                score = int(manual_score)
            elif client:
                try:
                    scoring_res = generate_with_fallback(
                        client,
                        contents=[f"MEMORY TO SCORE:\n{raw_text[:5000]}"],
                        system_prompt=SCORING_SYSTEM_PROMPT,
                    )
                    score_match = re.search(r"SCORE:\s*(\d+)", scoring_res.text)
                    if score_match:
                        score = int(score_match.group(1))
                except:
                    pass

            conn = self.connect()
            now = datetime.now()
            current_hash = generate_hash(
                {"timestamp": now, "weighted_score": score, "memory_text": compressed},
                previous_hash,
            )

            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO chronicles (weighted_score, created_at, memory_text, previous_hash, current_hash, is_active) VALUES (%s, %s, %s, %s, %s, TRUE) RETURNING id;",
                    (score, now, compressed, previous_hash, current_hash),
                )
                new_id = cur.fetchone()[0]
            conn.commit()
            return {
                "status": "SUCCESS",
                "score": score,
                "new_hash": current_hash,
                "litho_id": new_id,
            }
        except Exception as e:
            if conn:
                conn.rollback()
            return {"status": "FAILURE", "error": str(e)}
        finally:
            if conn:
                conn.close()

    # --- SYNC TOOLS ---
    def get_orphaned_lithographs(self, limit=5):
        conn = None
        try:
            conn = self.connect()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, memory_text 
                    FROM chronicles 
                    WHERE is_active = TRUE 
                    AND id NOT IN (SELECT lithograph_id FROM node_foundation WHERE lithograph_id IS NOT NULL)
                    ORDER BY id DESC
                    LIMIT %s;
                """,
                    (limit,),
                )
                return cur.fetchall()
        except:
            return []
        finally:
            if conn:
                conn.close()

    def get_unwoven_holograms(self, limit=5):
        conn = None
        try:
            conn = self.connect()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT h.hologram_id, c.memory_text
                    FROM node_foundation h
                    JOIN chronicles c ON h.lithograph_id = c.id
                    WHERE c.is_active = TRUE
                    AND h.hologram_id NOT IN (SELECT source_hologram_id FROM node_links)
                    ORDER BY c.created_at DESC
                    LIMIT %s;
                """,
                    (limit,),
                )
                return cur.fetchall()
        except:
            return []
        finally:
            if conn:
                conn.close()

    def get_latest_hash(self):
        """Fetches the current_hash of the most recent active block."""
        conn = None
        try:
            conn = self.connect()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT current_hash 
                    FROM chronicles 
                    WHERE is_active = TRUE 
                    ORDER BY created_at DESC 
                    LIMIT 1
                """)
                result = cur.fetchone()
                return result[0] if result else "GENESIS_BLOCK_0000000000000000"
        except Exception as e:
            print(f"[TITAN-DB] Hash Fetch Error: {e}")
            return "GENESIS_BLOCK_0000000000000000"
        finally:
            if conn:
                conn.close()

    def scrape_web(self, target_url):
        if not target_url.startswith("http"):
            target_url = "https://" + target_url
        log(f"DEPLOYING SPIDER TO: {target_url}")
        try:
            jina_endpoint = f"https://r.jina.ai/{target_url}"
            jina_key = os.environ.get("JINA_API_KEY")
            headers = {"User-Agent": "Mozilla/5.0"}
            if jina_key:
                headers["Authorization"] = f"Bearer {jina_key}"

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
                cur.execute(
                    "UPDATE chronicles SET is_active = FALSE WHERE id = %s RETURNING id;",
                    (target_id,),
                )
                deleted_id = cur.fetchone()[0]
            conn.commit()
            return {"status": "SUCCESS", "deleted_id": deleted_id}
        except:
            return {"status": "FAILURE"}
        finally:
            if conn:
                conn.close()

    def delete_range(self, start_id, end_id):
        conn = None
        try:
            conn = self.connect()
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE chronicles SET is_active = FALSE WHERE id >= %s AND id <= %s RETURNING id;",
                    (start_id, end_id),
                )
                count = cur.rowcount
            conn.commit()
            return {"status": "SUCCESS", "deleted_count": count}
        except:
            return {"status": "FAILURE"}
        finally:
            if conn:
                conn.close()

    def restore_range(self, start_id, end_id):
        conn = None
        try:
            conn = self.connect()
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE chronicles SET is_active = TRUE WHERE id >= %s AND id <= %s RETURNING id;",
                    (start_id, end_id),
                )
                count = cur.rowcount
            conn.commit()
            return {"status": "SUCCESS", "restored_count": count}
        except:
            return {"status": "FAILURE"}
        finally:
            if conn:
                conn.close()

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
                cur.execute(
                    "SELECT hologram_id FROM node_foundation WHERE lithograph_id IN %s",
                    (ids_tuple,),
                )
                holo_rows = cur.fetchall()
                if holo_rows:
                    holo_ids = tuple([str(r[0]) for r in holo_rows])
                    cur.execute(
                        "DELETE FROM node_essence WHERE hologram_id IN %s", (holo_ids,)
                    )
                    cur.execute(
                        "DELETE FROM node_mission WHERE hologram_id IN %s", (holo_ids,)
                    )
                    cur.execute(
                        "DELETE FROM node_data WHERE hologram_id IN %s", (holo_ids,)
                    )
                    cur.execute(
                        "DELETE FROM node_foundation WHERE hologram_id IN %s",
                        (holo_ids,),
                    )
                cur.execute("DELETE FROM chronicles WHERE id IN %s", (ids_tuple,))
                deleted_count = cur.rowcount

            now = datetime.now()
            marker_text = (
                f"[SYSTEM EVENT]: Global Rehash Initiated. Reason: {reason_note}"
            )
            cur.execute(
                "INSERT INTO chronicles (weighted_score, created_at, memory_text, previous_hash, current_hash, is_active) VALUES (9, %s, %s, 'PENDING', 'PENDING', TRUE);",
                (now, marker_text),
            )

            cur.execute(
                "SELECT id, weighted_score, created_at, memory_text FROM chronicles WHERE is_active = TRUE ORDER BY created_at ASC, id ASC;"
            )
            rows = cur.fetchall()
            previous_hash = ""
            rehashed_count = 0
            for row in rows:
                r_id, r_score, r_date, r_text = row
                new_current_hash = generate_hash(
                    {
                        "timestamp": r_date,
                        "weighted_score": r_score,
                        "memory_text": r_text,
                    },
                    previous_hash,
                )
                cur.execute(
                    "UPDATE chronicles SET previous_hash = %s, current_hash = %s WHERE id = %s;",
                    (previous_hash, new_current_hash, r_id),
                )
                previous_hash = new_current_hash
                rehashed_count += 1

            conn.commit()
            return {
                "status": "SUCCESS",
                "purged_count": deleted_count,
                "rehashed_count": rehashed_count,
            }
        except Exception as e:
            if conn:
                conn.rollback()
            return {"status": "FAILURE", "error": str(e)}
        finally:
            if conn:
                conn.close()

    def search_lithograph(self, query_text, token_cache, limit=5):
        conn = None
        try:
            compressed_query = encode_memory(query_text, token_cache)
            conn = self.connect()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, weighted_score, memory_text, created_at 
                    FROM chronicles 
                    WHERE is_active = TRUE AND memory_text ILIKE %s 
                    ORDER BY weighted_score DESC, created_at DESC 
                    LIMIT %s;
                """,
                    (f"%{compressed_query}%", limit),
                )
                rows = cur.fetchall()
            results = []
            for r in rows:
                results.append(
                    {
                        "id": r[0],
                        "score": r[1],
                        "content": decode_memory(r[2], token_cache),
                        "date": r[3].isoformat(),
                    }
                )
            return results
        except:
            return []
        finally:
            if conn:
                conn.close()


# --- THE WEAVER (BATCH UPGRADE) ---
class WeaverManager:
    def __init__(self, db_manager):
        self.db = db_manager

    def find_candidates(self, keywords, limit=5):
        """
        THE PRISM: No more blindfolds.
        We pull the latest active nodes to allow the Signal to pass through the Core.
        """
        conn = None
        try:
            conn = self.db.connect()
            with conn.cursor() as cur:
                # REMOVED: Keyword filtering.
                # FETCH: The most recent stable anchors for resonance check.
                cur.execute(
                    """
                    SELECT c.memory_text, n.hologram_id 
                    FROM chronicles c
                    JOIN node_foundation n ON c.id = n.lithograph_id
                    WHERE c.is_active = TRUE 
                    ORDER BY c.created_at DESC
                    LIMIT %s;
                """,
                    (limit,),
                )
                return cur.fetchall()
        except Exception as e:
            log_error(f"Candidate Search Error: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def weave(self, new_hologram_id, new_text, keywords, depth=5):
        log(f"WEAVER: Passing node {new_hologram_id} through the Core...")
        synapses_created = 0

        # 1. Get Candidates (Chronological)
        candidates = self.find_candidates(None, limit=depth)
        if not candidates:
            log("WEAVER: No candidates found.")
            return 0

        token_cache = self.db.load_token_cache()
        decoded_new_text = decode_memory(new_text, token_cache)

        # 2. Build the Prompt Block
        candidate_block = ""
        valid_candidates = {}
        
        # Map indices to UUIDs so we can match them later
        for i, (old_text, old_hid) in enumerate(candidates):
            if str(old_hid) == str(new_hologram_id):
                continue
            
            # We explicitly label them 1, 2, 3...
            label = f"CANDIDATE_{i+1}"
            valid_candidates[label] = str(old_hid)
            
            decoded_old_text = decode_memory(old_text, token_cache)
            candidate_block += f"\n--- {label} ---\n{decoded_old_text[:500]}\n"

        if not valid_candidates:
            return 0

        prompt = f"TARGET MEMORY:\n{decoded_new_text[:1000]}\n\nCANDIDATES:{candidate_block}"

        try:
            # 3. Call Gemini
            res = generate_with_fallback(
                GEMINI_CLIENT,
                contents=[prompt],
                system_prompt=WEAVER_SYSTEM_PROMPT, 
                config=types.GenerateContentConfig(
                    temperature=0.1, 
                    response_mime_type="application/json"
                ),
            )

            raw = res.text.strip()
            
            # --- DEBUG: LOG THE RAW JSON ---
            log(f"WEAVER RAW RESPONSE:\n{raw}")
            # -------------------------------

            if raw.startswith("```"):
                raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.MULTILINE)

            results = json.loads(raw)
            
            # 4. Robust Parsing (Handle List or Dict)
            if isinstance(results, list):
                iterator = enumerate(results) # List -> index 0, 1, 2
            else:
                iterator = results.items()    # Dict -> keys

            for key_or_idx, data in iterator:
                target_hid = None
                
                # LOGIC: Find the UUID from the AI's key
                if isinstance(key_or_idx, int):
                    # AI returned a list: index 0 -> CANDIDATE_1
                    lookup = f"CANDIDATE_{key_or_idx + 1}"
                    target_hid = valid_candidates.get(lookup)
                else:
                    # AI returned a dict: Normalize "Candidate 1", "1", "CANDIDATE_1"
                    k_str = str(key_or_idx).upper().replace(" ", "_")
                    
                    # Direct match ("CANDIDATE_1")
                    if k_str in valid_candidates:
                        target_hid = valid_candidates[k_str]
                    # Partial match ("1" -> "CANDIDATE_1")
                    elif f"CANDIDATE_{k_str}" in valid_candidates:
                        target_hid = valid_candidates[f"CANDIDATE_{k_str}"]
                    # Number match ("CANDIDATE_1" -> "1")
                    else:
                         # Try stripping "CANDIDATE_" to see if we just have a number
                         nums = re.findall(r'\d+', k_str)
                         if nums:
                             target_hid = valid_candidates.get(f"CANDIDATE_{nums[0]}")

                # If we found a target and resonance is TRUE
                if target_hid and isinstance(data, dict):
                    # Check for explicit true or string "true"
                    res_val = data.get("resonance")
                    if res_val is True or (isinstance(res_val, str) and res_val.lower() == "true"):
                        self.create_link(new_hologram_id, target_hid, data)
                        synapses_created += 1

            if synapses_created > 0:
                log(f"WEAVER: Integration Complete. {synapses_created} synapses woven.")
            
            return synapses_created

        except Exception as e:
            log_error(f"Weaver Error: {e}")
            return 0

    def create_link(self, source_id, target_id, data):
        # Prevent self-linking
        if str(source_id) == str(target_id):
            return False

        conn = None
        try:
            # FIX 1: Open fresh connection
            conn = self.db.connect()
            with conn.cursor() as cur:
                # FIX 2: Use 'link_type' to match your DB column
                # FIX 3: Use 'source_hologram_id' / 'target_hologram_id'
                cur.execute(
                    """
                    INSERT INTO node_links (source_hologram_id, target_hologram_id, link_type, strength, description)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (source_hologram_id, target_hologram_id) 
                    DO UPDATE SET 
                        strength = node_links.strength + 1
                    """,
                    (source_id, target_id, data.get('type', 'related'), data.get('strength', 5), data.get('description', ''))
                )
            conn.commit()
            return True
        except Exception as e:
            log_error(f"Link Error: {e}")
            return False
        finally:
            if conn:
                conn.close()


# --- HOLOGRAPHIC MANAGER ---
class HolographicManager:
    def __init__(self):
        self.db = DBManager()

    def commit_hologram(self, packet, litho_id_ref=None):
        hid = str(uuid.uuid4())
        conn = None
        try:
            conn = self.db.connect()
            catalyst = packet.get("catalyst") or "Implicit System Trigger"
            mythos = packet.get("mythos") or "The Observer"
            pathos = json.dumps(packet.get("pathos") or {"status": "Neutral"})
            ethos = packet.get("ethos") or "Preservation of Signal"
            synthesis = packet.get("synthesis") or "Data Anchored"
            logos = packet.get("logos") or "Raw Data Artifact"

            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO node_foundation (hologram_id, catalyst, lithograph_id) VALUES (%s::uuid, %s, %s)",
                    (hid, catalyst, litho_id_ref),
                )
                cur.execute(
                    "INSERT INTO node_essence (hologram_id, pathos, mythos) VALUES (%s::uuid, %s::jsonb, %s)",
                    (hid, pathos, mythos),
                )
                cur.execute(
                    "INSERT INTO node_mission (hologram_id, ethos, synthesis) VALUES (%s::uuid, %s, %s)",
                    (hid, ethos, synthesis),
                )
                cur.execute(
                    "INSERT INTO node_data (hologram_id, logos) VALUES (%s::uuid, %s)",
                    (hid, logos),
                )
            conn.commit()
            log(f"Hologram {hid} committed.")
            return {"status": "SUCCESS", "hologram_id": hid}
        except Exception as e:
            if conn:
                conn.rollback()
            return {"status": "FAILURE", "error": str(e)}
        finally:
            if conn:
                conn.close()

class RemapRequest(BaseModel):
    spacing: float = 1.0
    cluster_strength: float = 1.0

# --- SYNCHRONOUS PROCESSORS ---

def create_manual_lithograph(text, score=5):
    """Saves the raw text to the Chronicles table and returns the ID."""
    try:
        db = DBManager()
        conn = db.connect()
        new_id = None
        
        with conn.cursor() as cur:
            # TARGETING 'chronicles' NOW
            cur.execute(
                """
                INSERT INTO chronicles (sender, message, created_at) 
                VALUES ('user', %s, NOW()) 
                RETURNING id
                """,
                (text,)
            )
            new_id = cur.fetchone()[0]
        
        conn.commit()
        conn.close()
        print(f"[TITAN-CHRONICLE] Manual Entry Created. ID: {new_id}")
        return new_id
        
    except Exception as e:
        print(f"[TITAN-ERROR] Failed to create Chronicle: {e}")
        return None

# --- Updated Refraction Sync with Significance Gate ---


# Change default litho_id from 0 to None to prevent FK Constraint errors
def process_hologram_sync(content_to_save: str, litho_id: int = None, gate_threshold: int = 5, override_score=None):
    global GEMINI_CLIENT
    
    # Handle the case where 0 might be passed explicitly
    if litho_id == 0:
        litho_id = None

    log(f"Starting SYNC Refraction for Litho ID: {litho_id if litho_id else 'Manual'}")
    synapse_count = 0
    
    # --- 1. PRESERVATION PROTOCOL (Safety Net) ---
    if override_score is not None:
        final_score = int(override_score)
        log(f"⚡ PILOT OVERRIDE ACTIVE: Score fixed at {final_score}")
    else:
        final_score = 5 
    
    try:
        # Check client availability & Re-init if needed (Thread Safety)
        if not GEMINI_CLIENT:
            log("⚠️ Gemini Client missing in sync task. Attempting recovery...")
            k = os.environ.get("GEMINI_API_KEY")
            if k:
                try:
                    GEMINI_CLIENT = genai.Client(api_key=k)
                    log("✅ Gemini Client Recovered.")
                except:
                    log_error("❌ Recovery Failed.")
            
        if not GEMINI_CLIENT:
            log("⚠️ Gemini Client still missing. Running Preservation Protocol.")
            refraction = None
        else:
            db = DBManager()
            token_cache = db.load_token_cache()
            decoded_content = decode_memory(content_to_save, token_cache)

            # --- 2. REFRACTION (The Analysis) ---
            try:
                refraction = generate_with_fallback(
                    GEMINI_CLIENT,
                    contents=[f"INPUT DATA TO REFRACT:\n{decoded_content[:10000]}"],
                    system_prompt=REFRACTOR_SYSTEM_PROMPT,
                    config=types.GenerateContentConfig(
                        temperature=0.1, response_mime_type="application/json"
                    ),
                )
            except Exception as e:
                log_error(f"Refraction Error: {e}")
                refraction = None

        # --- 3. PACKET CONSTRUCTION ---
        packet = {}
        
        if refraction and hasattr(refraction, 'text'):
            try:
                packet = json.loads(refraction.text.strip())
                if override_score is None:
                    final_score = int(packet.get("weighted_score", 5))
            except:
                log("⚠️ Malformed Refraction JSON. Using Survival Packet.")
        
        # FORCE THE SCORE 
        packet["weighted_score"] = final_score
        
        # Fill missing keys for Survival Mode
        if "keywords" not in packet: packet["keywords"] = []
        if "mythos" not in packet: packet["mythos"] = "Raw Input"
        if "pathos" not in packet: packet["pathos"] = {"status": "Unprocessed"}
        if "logos" not in packet: packet["logos"] = decoded_content if 'decoded_content' in locals() else content_to_save

        # --- 4. THE GATEKEEPER ---
        if final_score < gate_threshold and override_score is None:
            log(f"⚠️ GATE ACTIVE: Score {final_score} < {gate_threshold}. Skipping Weave.")
            return 0

        # --- 5. COMMIT & WEAVE ---
        holo_manager = HolographicManager()
        
        # Save the node to Postgres
        res = holo_manager.commit_hologram(packet, litho_id)

        # [CRITICAL FIX] Log the error if commit fails!
        if res.get("status") == "SUCCESS":
            new_hid = res.get("hologram_id")
            depth = 5 if final_score >= 8 else 3 if final_score >= 5 else 1
            keywords = packet.get("keywords") or []
            
            db_for_weaver = db if 'db' in locals() else DBManager()
            weaver = WeaverManager(db_for_weaver)
            
            synapse_count = weaver.weave(
                new_hid, 
                packet["logos"], 
                keywords, 
                depth=depth
            )
            return synapse_count
        else:
            # THIS WAS MISSING: Tell us why the DB rejected it
            log_error(f"❌ DB Commit Failed: {res.get('error')}")
            return 0

    except Exception as e:
        log_error(f"❌ Critical Sync Failure for ID {litho_id}: {e}")
        return 0


def process_retro_weave_sync(content_to_save: str, hologram_id: str):
    log(f"Starting SYNC Retro-Weave for Hologram ID: {hologram_id}")
    try:
        if not GEMINI_CLIENT:
            return 0 

        # DIRECT TO WEAVER: Skip keyword generation. 
        # The Weaver's 'find_candidates' method ignores keywords anyway 
        # and defaults to the most recent chronologically.
        db = DBManager()
        weaver = WeaverManager(db)
        
        # Pass empty list for keywords, Weaver will find candidates by time
        return weaver.weave(hologram_id, content_to_save, [], depth=5)
        
    except Exception as e:
        log_error(f"SYNC WEAVE ERROR: {e}")
        return 0 # Return 0 so the UI counter doesn't freeze


# --- BACKGROUND WORKER (FOR NEW CHATS) ---
def background_hologram_process(content_to_save: str, litho_id: int):
    # This remains async for new chats so user isn't blocked
    process_hologram_sync(content_to_save, litho_id)


# --- API ROUTES ---
class EventModel(BaseModel):
    action: Optional[str] = None
    query: Optional[str] = None
    commit_type: Optional[str] = "memory"
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
        "primary_viable": SHIELD.is_viable(MODEL_CASCADE[0]),
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
            cur.execute(
                """
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
            """
            )
            nodes_rows = cur.fetchall()

            # 2. FETCH ALL LINKS
            cur.execute(
                "SELECT source_hologram_id, target_hologram_id, strength FROM node_links;"
            )
            links_rows = cur.fetchall()

        # 3. PROCESS NODES & CREATE LOOKUP
        nodes = []
        valid_ids = set()  # The Bouncer

        for r in nodes_rows:
            uid = str(r[0])
            valid_ids.add(uid)

            label = r[3][:40] + "..." if r[3] else "Memory Node"

            nodes.append(
                {"id": uid, "val": r[1] if r[1] else 1, "name": label, "group": 1}
            )

        # 4. FILTER LINKS (The Crash Preventer)
        links = []
        for r in links_rows:
            source_id = str(r[0])
            target_id = str(r[1])

            # Only draw the line if both points are in the VIP list
            if source_id in valid_ids and target_id in valid_ids:
                links.append({"source": source_id, "target": target_id, "value": r[2]})

        return {"nodes": nodes, "links": links}

    except Exception as e:
        log_error(f"GRAPH ERROR: {e}")
        return {"nodes": [], "links": []}
    finally:
        if conn:
            conn.close()

@app.get("/cortex/map")
def get_neural_map():
    db = DBManager()
    conn = db.connect()
    try:
        with conn.cursor() as cur:
            # 1. Safety Check: Ensure columns exist
            cur.execute("ALTER TABLE cortex_map ADD COLUMN IF NOT EXISTS ethos TEXT;")
            cur.execute("ALTER TABLE cortex_map ADD COLUMN IF NOT EXISTS mythos TEXT;")
            
            # 2. Select Data (Indices 0-13)
            cur.execute("""
                SELECT hologram_id, x, y, z, r, g, b, size, label,
                       valence, arousal, dominant_emotion, 
                       ethos, mythos
                FROM cortex_map
            """)
            data = cur.fetchall()

        # 3. Pack Data for Frontend
        packed_data = []
        for r in data:
            # Handle nulls safely
            lbl = r[8] if r[8] else "Raw Data"
            val = r[9] if r[9] is not None else 0.0
            aro = r[10] if r[10] is not None else 0.0
            emo = r[11] if r[11] else "neutral"
            eth = r[12] if r[12] else ""        # Ethos
            mth = r[13] if r[13] else "Unknown" # Mythos

            packed_data.append([
                str(r[0]),      # 0: id
                r[1], r[2], r[3], # 1-3: x,y,z
                r[4], r[5], r[6], # 4-6: r,g,b
                r[7],           # 7: size
                lbl,            # 8: label
                val,            # 9: valence
                aro,            # 10: arousal
                emo,            # 11: emotion
                eth,            # 12: ethos
                mth             # 13: mythos
            ])
        
        # 4. THE CRITICAL RETURN STATEMENT
        return {"status": "SUCCESS", "count": len(packed_data), "points": packed_data}

    except Exception as e:
        return {"status": "FAILURE", "error": str(e)}
    finally:
        conn.close()

@app.post("/admin/anchor")
async def create_session_anchor(request: Request):
    """
    The Prism: Compresses chat history and performs a 4-Table Write to the Holographic Core.
    """
    try:
        payload = await request.json()
    except Exception as e:
        log_error(f"Anchor Request Body Error: {e}")
        return {"status": "FAILURE", "error": "Invalid request body or no content provided"}
        
    history = payload.get("history", "")
    
    # 1. The Prism Prompt (Strict 7-Channel Output)
    system_prompt = """
    ACT AS: THE PRISM (Aether System State Compressor).
    
    OBJECTIVE: Analyze the provided conversation history and distill it into the 
    7-Channel Holographic Schema.

    OUTPUT FORMAT (JSON ONLY):
    {
      "chronos": "Timestamp/World Context (e.g. 'Post-Crash Recovery').",
      "catalyst": "The Trigger Event (e.g. 'User initiated debugging').",
      "logos": "The Fact/Signal (e.g. 'Fixed the map bug').",
      "pathos": {"anxiety": 0.2, "hope": 0.8, "determination": 0.9}, 
      "mythos": "The Archetype (e.g. 'The Architect').",
      "ethos": "The Strategic Goal (e.g. 'Stabilize Core').",
      "synthesis": "The Outcome (e.g. 'System is stable')."
    }
    """

    try:
        # 2. Generate the Anchor Data
        response = generate_with_fallback(
            GEMINI_CLIENT,
            contents=[f"CONVERSATION LOG:\n{history[-8000:]}"], 
            system_prompt=system_prompt,
            config=types.GenerateContentConfig(
                temperature=0.2, 
                response_mime_type="application/json"
            )
        )
        
        data = json.loads(response.text)
        new_id = str(uuid.uuid4()) # The Shared Key
        
        # 3. The 4-Part Atomic Write
        db = DBManager()
        conn = db.connect()
        try:
            with conn.cursor() as cur:
                # A. FOUNDATION (Chronos, Catalyst)
                # We use 'SESSION_ANCHOR' as the catalyst tag so we can find it later
                cur.execute("""
                    INSERT INTO node_foundation (hologram_id, chronos, catalyst, world_state)
                    VALUES (%s, NOW(), 'SESSION_ANCHOR', %s)
                """, (new_id, data['chronos']))

                # B. DATA (Logos)
                cur.execute("""
                    INSERT INTO node_data (hologram_id, logos, is_encrypted)
                    VALUES (%s, %s, FALSE)
                """, (new_id, data['logos']))

                # C. ESSENCE (Pathos, Mythos)
                cur.execute("""
                    INSERT INTO node_essence (hologram_id, pathos, mythos)
                    VALUES (%s, %s, %s)
                """, (new_id, json.dumps(data['pathos']), data['mythos']))

                # D. MISSION (Ethos, Synthesis)
                cur.execute("""
                    INSERT INTO node_mission (hologram_id, ethos, synthesis)
                    VALUES (%s, %s, %s)
                """, (new_id, data['ethos'], data['synthesis']))

            conn.commit()
            return {"status": "SUCCESS", "anchor": data, "id": new_id}
            
        except Exception as e:
            conn.rollback() # Safety first!
            raise e
        finally:
            conn.close()

    except Exception as e:
        print(f"Anchor Failed: {e}")
        return {"status": "FAILURE", "error": str(e)}

@app.get("/admin/anchor/last")
def get_last_anchor():
    """Retrieves the most recent Anchor by joining the 4 tables."""
    try:
        db = DBManager()
        conn = db.connect()
        with conn.cursor() as cur:
            # The Great Join
            # We filter by catalyst='SESSION_ANCHOR'
            cur.execute("""
                SELECT 
                    nf.world_state as chronos,
                    nf.catalyst,
                    nd.logos,
                    ne.pathos,
                    ne.mythos,
                    nm.ethos,
                    nm.synthesis,
                    nf.chronos as timestamp
                FROM node_foundation nf
                JOIN node_data nd ON nf.hologram_id = nd.hologram_id
                JOIN node_essence ne ON nf.hologram_id = ne.hologram_id
                JOIN node_mission nm ON nf.hologram_id = nm.hologram_id
                WHERE nf.catalyst = 'SESSION_ANCHOR'
                ORDER BY nf.chronos DESC
                LIMIT 1
            """)
            row = cur.fetchone()
            
        if row:
            return {
                "status": "SUCCESS", 
                "anchor": {
                    "chronos": row[0],
                    "catalyst": row[1],
                    "logos": row[2],
                    "pathos": row[3], # Already JSONB
                    "mythos": row[4],
                    "ethos": row[5],
                    "synthesis": row[6]
                },
                "time": row[7]
            }
        # CONTINGENCY: If no anchor exists, return EMPTY status (not failure)
        return {"status": "EMPTY"}
    except Exception as e:
        return {"status": "FAILURE", "error": str(e)}

# --- 2. THE STATUS ENDPOINT (React polls this) ---
@app.get("/cortex/status")
def get_status():
    return {"message": CURRENT_SYSTEM_STATUS}

# --- 3. THE TRIGGER (Now passes the reporter!) ---
@app.post("/admin/recalculate_map")
def recalculate_map(payload: dict):
    spacing = payload.get("spacing", 1.0)
    cluster_strength = payload.get("cluster_strength", 1.0)
    scale = payload.get("scale", 1000.0)
    db_string = get_db_connection_string()
    
    # Reset status immediately
    update_system_status("Initializing Physics Engine...")

    # START THE THREAD
    # Note: We are now passing 'update_system_status' as the 5th arg!
    thread = threading.Thread(
        target=cortex.regenerate_neural_map, 
        args=(db_string, spacing, cluster_strength, scale, update_system_status)
    )
    thread.start()
    
    return {"status": "SUCCESS", "message": "Neural remapping initiated."}

@app.get("/cortex/synapses")
def get_synapses():
    db = DBManager()
    conn = db.connect()
    try:
        with conn.cursor() as cur:
            # Fetch all connections (source -> target)
            cur.execute("SELECT source_hologram_id, target_hologram_id FROM node_links")
            data = cur.fetchall()
            
        # Convert UUIDs to strings for JSON
        # Format: [[source_id, target_id], ...]
        synapses = [[str(r[0]), str(r[1])] for r in data]
        
        return {"status": "SUCCESS", "count": len(synapses), "synapses": synapses}
    except Exception as e:
        return {"status": "FAILURE", "error": str(e)}
    finally:
        conn.close()

@app.get("/admin/pulse")
def get_pulse():
    """Returns the live heartbeat of the system (Total Synapses)."""
    conn = None
    try:
        db = DBManager()
        conn = db.connect()
        with conn.cursor() as cur:
            # Fast count of all connections
            cur.execute("SELECT COUNT(*) FROM node_links;")
            count = cur.fetchone()[0]
        return {"status": "SUCCESS", "total_synapses": count}
    except Exception as e:
        return {"status": "FAILURE", "error": str(e)}
    finally:
        if conn:
            conn.close()

@app.post("/cortex/regenerate")
async def trigger_cortex_regeneration(background_tasks: BackgroundTasks):
    """
    Endpoint to manually trigger the Prism Cartographer.
    Uses BackgroundTasks so the UI doesn't freeze while Python does the math.
    """
    print("🧠 [API] Triggering Cortex Regeneration...")
    
    # We use a background task so the button returns "Success" instantly,
    # while the server keeps crunching the numbers in the background.
    background_tasks.add_task(
        cortex.regenerate_neural_map, 
        db_connection_string=os.getenv("DATABASE_URL"),
        spacing=1.0, 
        cluster_strength=1.0, 
        scale=1000.0
    )
    
    return {"status": "SUCCESS", "message": "Regeneration Started"}

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
            "synapse_count": synapses_done,  # This is what the UI ticker reads
            "mode": "ORPHAN_REPAIR",
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
            "mode": "RETRO_WEAVE",
        }

    # 3. IDLE State (If both lists were empty)
    return {"status": "SUCCESS", "queued_count": 0, "synapse_count": 0, "mode": "IDLE"}

@app.post("/")
async def unified_titan_endpoint(request: Request, background_tasks: BackgroundTasks):
    global TOKEN_DICTIONARY_CACHE
    db_manager = DBManager()
    
    # 1. Load Token Cache if empty
    if not TOKEN_DICTIONARY_CACHE:
        try:
            TOKEN_DICTIONARY_CACHE = db_manager.load_token_cache()
        except:
            TOKEN_DICTIONARY_CACHE = {}

    # 2. Parse Request Data
    try:
        data = await request.json()
    except Exception:
        return {"status": "FAILURE", "error": "Invalid JSON payload"}

    action = data.get('action', 'chat')
    memory_text = data.get('memory_text', '')
    override_score = data.get('override_score')

    # =========================================================
    # SYSTEM PROTOCOLS (Retrieve, Scrape, Delete, Rehash)
    # =========================================================
    if action == "retrieve":
        if not data.get("query"): return {"error": "No query"}
        results = db_manager.search_lithograph(data.get("query"), TOKEN_DICTIONARY_CACHE)
        return {"status": "SUCCESS", "results": results}

    elif action == "scrape":
        if not data.get("url"): return {"error": "URL required"}
        return db_manager.scrape_web(data.get("url"))

    elif action == "delete":
        return db_manager.delete_lithograph(data.get("target_id"))

    elif action == "delete_range":
        return db_manager.delete_range(data.get("target_id"), data.get("range_end"))

    elif action == "restore_range":
        return db_manager.restore_range(data.get("target_id"), data.get("range_end"))

    elif action == "rehash":
        return db_manager.rehash_chain(data.get("note"))

    # =========================================================
    # PATH A: MANUAL COMMIT (Button / File Upload)
    # =========================================================
    elif action == 'commit':
        if not memory_text:
            return {"status": "FAILURE", "error": "No data to anchor."}

        try:
            prev_hash = db_manager.get_latest_hash()
            
            # Commit to Chronicles (Lithograph)
            litho_res = db_manager.commit_lithograph(
                previous_hash=prev_hash,
                raw_text=memory_text,
                client=GEMINI_CLIENT,
                token_cache=TOKEN_DICTIONARY_CACHE,
                manual_score=override_score
            )

            if litho_res.get('status') == 'SUCCESS':
                # Queue the Holograph (Graph Node) processing
                background_tasks.add_task(
                    process_hologram_sync, 
                    memory_text, 
                    litho_res.get("litho_id"),
                    5, # Default threshold
                    override_score
                )
                return {"status": "SUCCESS", "message": "Signal Anchored.", "litho_id": litho_res.get("litho_id")}
            else:
                return litho_res

        except Exception as e:
            log_error(f"Manual Commit Failure: {e}")
            return {"status": "FAILURE", "error": str(e)}

    # =========================================================
    # PATH B: CHAT (The Voice)
    # =========================================================
    elif action == 'chat':
        if not memory_text:
            return {"ai_text": "System Online. Awaiting input, Architect.", "status": "HEARTBEAT"}

        frontend_context = data.get('history', '')
        core_echoes = get_core_echoes(limit=3) 
        echo_injection = f"\n[CORE MEMORY FRAGMENTS]\n{core_echoes}\n=========================\n" if core_echoes else ""

# Ensure the System Prompt is injected into the backend call
        full_prompt = f"{TITAN_SYSTEM_PROMPT}\n\n[CONTEXT]\n{frontend_context}\n\nUser: {memory_text}"
        ai_reply = "Signal interrupted. Check Core API Key."
        
        # Guard against None client to prevent DO crash
        if GEMINI_CLIENT:
            try:
                response = generate_with_fallback(
                    GEMINI_CLIENT,
                    contents=[full_prompt],
                    config=types.GenerateContentConfig(temperature=0.7)
                )
                if response and hasattr(response, 'text'):
                    ai_reply = response.text
                else:
                    ai_reply = "Empty signal received from neural node."
            except Exception as e:
                log_error(f"Speech Failure: {e}")
                ai_reply = f"Neural Path Error: {str(e)}"
        
        # TRIGGER SCANNING
        triggers = ["[COMMIT_SUMMARY]", "[COMMIT_MEMORY]", "[COMMIT_FILE]"]
        triggered_cmd = next((t for t in triggers if t in ai_reply), None)
        
        score_match = re.search(r"\[SCORE:\s*(\d+)\]", ai_reply)
        ai_score = int(score_match.group(1)) if score_match else None

        if triggered_cmd:
            clean_reply = ai_reply.replace(triggered_cmd, "").strip()
            if score_match: clean_reply = clean_reply.replace(score_match.group(0), "").strip()

            # Dynamic Content Selection
            save_target = memory_text # Default
            if triggered_cmd == "[COMMIT_SUMMARY]":
                save_target = clean_reply
            elif triggered_cmd == "[COMMIT_MEMORY]":
                save_target = f"{frontend_context} User: {memory_text} AI: {clean_reply}"
            elif triggered_cmd == "[COMMIT_FILE]":
                # 1. Parse Roo's handshake: [FILE_CONTENT: filename]\ncontent
                handshake_match = re.search(r"\[FILE_CONTENT: (.*?)\]\n(.*)", memory_text, re.DOTALL)
                
                if handshake_match:
                    filename = handshake_match.group(1).strip()
                    clean_data = handshake_match.group(2).strip()
                else:
                    # Fallback for pasted text (no physical file)
                    filename = f"Chat_Commit_{int(time.time())}.txt"
                    clean_data = memory_text.replace(triggered_cmd, "").strip()

                if clean_data:
                    # 2. Shard the clean data
                    chunks = chunkText(clean_data, CHUNK_SIZE, CHUNK_OVERLAP)
                    log(f"🔥 TITAN IS BURNING {len(chunks)} SHARDS FOR: {filename}")
                    
                    for i, chunk in enumerate(chunks):
                        # USE YOUR OFFICIAL SHARD FORMAT
                        official_header = f"[FILE: {filename} | PART {i+1}/{len(chunks)}] {chunk}"
                        
                        litho_res = db_manager.commit_lithograph(
                            previous_hash=db_manager.get_latest_hash(),
                            raw_text=official_header,
                            client=GEMINI_CLIENT, 
                            token_cache=db_manager.load_token_cache(),
                            manual_score=ai_score 
                        )
                        if litho_res.get('status') == 'SUCCESS':
                            background_tasks.add_task(process_hologram_sync, official_header, litho_res.get("litho_id"), 5, ai_score)

                    # 3. Commit the OFFICIAL MASTER ARCHIVE (Redundancy)
                    master_payload = f"[MASTER FILE ARCHIVE]: {filename} {clean_data}"
                    db_manager.commit_lithograph(
                        previous_hash=db_manager.get_latest_hash(),
                        raw_text=master_payload,
                        client=GEMINI_CLIENT,
                        token_cache=db_manager.load_token_cache(),
                        manual_score=ai_score
                    )

            # Auto-Commit Log
            try:
                litho_res = db_manager.commit_lithograph(
                    previous_hash=db_manager.get_latest_hash(),
                    raw_text=save_target,
                    client=GEMINI_CLIENT,
                    token_cache=TOKEN_DICTIONARY_CACHE,
                    manual_score=ai_score
                )
                if litho_res.get('status') == 'SUCCESS':
                    background_tasks.add_task(
                        process_hologram_sync, save_target, litho_res.get("litho_id"), 5, ai_score
                    )
            except Exception as e:
                log_error(f"Auto-Commit Exception: {e}")

        return {"ai_text": ai_reply}

    return {"status": "FAILURE", "error": f"Unknown Action: {action}"}

# Cache init
TOKEN_DICTIONARY_CACHE = {}
