import hashlib
import json
import os
import psycopg2
import re
import uuid
import asyncio 
import urllib.parse 
from datetime import datetime
from google import genai
from flask import Flask, request, jsonify 
from flask_cors import CORS
import traceback # Added for detailed error logging

# --- FLASK APP INSTANCE ---
app = Flask(__name__)
CORS(app) 

# --- GLOBAL VARIABLES & PROMPTS ---
TOKEN_DICTIONARY_CACHE = {}
GEMINI_CLIENT = None 

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

# --- SECURE CLIENT INITIALIZATION ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("FATAL CONFIG ERROR: GEMINI_API_KEY not found.")
else:
    try:
        GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"FATAL INITIALIZATION ERROR: {e}")

# --- UTILITIES ---

def generate_hash(memory_data, previous_hash_string):
    if isinstance(memory_data.get("timestamp"), datetime):
        memory_data["timestamp"] = memory_data["timestamp"].isoformat()
    data_block_string = json.dumps(memory_data, sort_keys=True)
    raw_content = previous_hash_string + data_block_string
    return hashlib.sha256(raw_content.encode('utf-8')).hexdigest()

def get_weighted_score(memory_text, client, token_cache):
    override_match = re.search(r"\[SCORE:\s*([0-9])\]", memory_text, re.IGNORECASE)
    if override_match: return int(override_match.group(1))
    if client is None: return 5
    try:
        decoded_text = decode_memory(memory_text, token_cache) if token_cache else memory_text
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=[SCORING_SYSTEM_PROMPT + decoded_text],
            config={"temperature": 0.0} 
        )
        match = re.search(r"SCORE:\s*([0-9])", response.text.strip(), re.IGNORECASE)
        return int(match.group(1)) if match else 5
    except: return 5

def encode_memory(raw_text, token_map):
    if not token_map: return raw_text
    compressed_text = raw_text
    sorted_tokens = sorted(token_map.items(), key=lambda item: len(item[0]), reverse=True)
    for phrase, hash_code in sorted_tokens:
        if phrase in compressed_text:
            compressed_text = compressed_text.replace(phrase, hash_code)
    return compressed_text

def decode_memory(compressed_text, token_map):
    if not token_map: return compressed_text
    decompressed_text = compressed_text
    decode_map = {v: k for k, v in token_map.items()}
    sorted_hashes = sorted(decode_map.keys(), key=len, reverse=True)
    for hash_code in sorted_hashes:
        if hash_code in decompressed_text:
            decompressed_text = decompressed_text.replace(hash_code, decode_map[hash_code])
    return decompressed_text

# --- CORE MANAGERS ---

def get_db_connection_string():
    password = urllib.parse.quote_plus(os.environ.get("DB_PASSWORD", ""))
    return f"postgresql://{os.environ.get('DB_USER')}:{password}@{os.environ.get('DB_HOST')}:{os.environ.get('DB_PORT', '6543')}/{os.environ.get('DB_NAME')}?sslmode=require"

class DBManager:
    def __init__(self):
        self.connection_string = get_db_connection_string()
        self.connection = None

    def connect(self):
        if self.connection is None:
            self.connection = psycopg2.connect(self.connection_string)
            self.connection.autocommit = False  
        return self.connection

    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None

    def load_token_cache(self):
        token_cache = {}
        try:
            conn = self.connect()
            with conn.cursor() as cur:
                cur.execute("SELECT english_phrase, hash_code FROM token_dictionary;")
                for p, c in cur.fetchall(): token_cache[p.strip()] = c.strip()
            return token_cache
        except Exception as e:
            print(f"Cache Error: {e}")
            return {}
        finally: self.close()

    def commit_lithograph(self, previous_hash, raw_text, client, token_cache, manual_score=None):
        try:
            compressed = encode_memory(raw_text, token_cache)
            score = int(manual_score) if manual_score is not None else get_weighted_score(raw_text, client, token_cache)
            conn = self.connect()
            now = datetime.now()
            current_hash = generate_hash({"timestamp": now, "weighted_score": score, "memory_text": compressed}, previous_hash)
            
            with conn.cursor() as cur:
                cur.execute("INSERT INTO chronicles (weighted_score, created_at, memory_text, previous_hash, current_hash) VALUES (%s, %s, %s, %s, %s) RETURNING id;", 
                            (score, now, compressed, previous_hash, current_hash))
                new_id = cur.fetchone()[0]
            conn.commit()
            return {"status": "SUCCESS", "score": score, "new_hash": current_hash, "litho_id": new_id}
        except Exception as e:
            if self.connection: self.connection.rollback()
            return {"status": "FAILURE", "error": str(e)}

    def search_lithograph(self, query_text, token_cache, limit=5):
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
            print(f"Search Error: {e}")
            return []

# --- HOLOGRAPHIC MANAGER (ASYNC BRIDGE) ---

class HolographicManager:
    # We do NOT pass db_manager here anymore to avoid shared connection issues in async threads
    def __init__(self):
        pass

    async def commit_hologram(self, packet, litho_id_ref=None):
        # Generate ID
        hid = str(uuid.uuid4())
        
        # --- DEBUG LOGGING (Check DigitalOcean Logs for this!) ---
        print(f"TITAN DEBUG: Attempting to write Hologram ID: {hid}", flush=True)
        print(f"TITAN DEBUG: Packet Data: {json.dumps(packet)}", flush=True)

        # Create connection
        local_db = DBManager()
        
        try:
            conn = local_db.connect()
            
            # --- STABILIZER: DEFINE DEFAULTS ---
            catalyst = packet.get('catalyst') or "Implicit System Trigger"
            mythos = packet.get('mythos') or "The Observer"
            pathos = json.dumps(packet.get('pathos') or {"status": "Neutral"})
            ethos = packet.get('ethos') or "Preservation of Signal"
            synthesis = packet.get('synthesis') or "Data Anchored"
            logos = packet.get('logos') or "Raw Data Artifact"

            with conn.cursor() as cur:
                # 1. THE REAL WRITE (With explicit UUID casting)
                cur.execute(
                    "INSERT INTO node_foundation (hologram_id, catalyst) VALUES (%s::uuid, %s)", 
                    (hid, catalyst)
                )
                cur.execute(
                    "INSERT INTO node_essence (hologram_id, pathos, mythos) VALUES (%s::uuid, %s, %s)", 
                    (hid, pathos, mythos)
                )
                cur.execute(
                    "INSERT INTO node_mission (hologram_id, ethos, synthesis) VALUES (%s::uuid, %s, %s)", 
                    (hid, ethos, synthesis)
                )
                cur.execute(
                    "INSERT INTO node_data (hologram_id, logos) VALUES (%s::uuid, %s)", 
                    (hid, logos)
                )
                
                # 2. THE HARDCODED TEST (To verify Python connectivity)
                test_id = str(uuid.uuid4())
                cur.execute(
                    "INSERT INTO node_data (hologram_id, logos) VALUES (%s::uuid, %s)", 
                    (test_id, "PYTHON CONNECTION PROBE - IF YOU SEE THIS, PYTHON IS WORKING")
                )

            conn.commit()
            print(f"TITAN LOG: COMMIT EXECUTION FINISHED for {hid}", flush=True)
            return {"status": "SUCCESS", "hologram_id": hid}

        except Exception as e:
            if local_db.connection: local_db.connection.rollback()
            error_details = traceback.format_exc()
            print(f"TITAN ERROR (Hologram Reject): {error_details}", flush=True) 
            return {"status": "FAILURE", "error": str(e), "details": error_details}
        finally:
            local_db.close()

# --- LOGIC ROUTER ---

def application_logic(event):
    global TOKEN_DICTIONARY_CACHE

    if not TOKEN_DICTIONARY_CACHE:
        db = DBManager()
        try:
            TOKEN_DICTIONARY_CACHE = db.load_token_cache()
        except:
            pass

    db_manager = DBManager()
    db_manager.connect()

    try:
        action = event.get('action')
        
        # 1. RETRIEVE
        if action == 'retrieve':
            query_text = event.get('query')
            if not query_text: return {'statusCode': 400, 'body': json.dumps({'error': 'No query'})}
            results = db_manager.search_lithograph(query_text, TOKEN_DICTIONARY_CACHE)
            return {'statusCode': 200, 'body': json.dumps({'status': 'SUCCESS', 'results': results})}

        # 2. COMMIT
        commit_type = event.get('commit_type', 'memory')
        new_text = event.get('memory_text')
        
        if not new_text: return {'statusCode': 200, 'body': json.dumps({'status': 'HEARTBEAT'})}

        # Summarization Logic
        content_to_save = new_text
        if commit_type == 'summary':
            try:
                summary_res = GEMINI_CLIENT.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[f"Summarize this interaction for the Lithographic Core (Keep it dense and factual): {new_text}"]
                )
                content_to_save = summary_res.text
            except:
                pass 

        # Lithographic Commit (Main Thread)
        prev_hash = ''
        try:
            with db_manager.connect().cursor() as cur:
                cur.execute("SELECT current_hash FROM chronicles ORDER BY id DESC LIMIT 1;")
                res = cur.fetchone()
                prev_hash = res[0].strip() if res else ''
        except: pass
            
        litho_res = db_manager.commit_lithograph(prev_hash, content_to_save, GEMINI_CLIENT, TOKEN_DICTIONARY_CACHE, event.get('override_score'))

        # 3. Holographic Refraction (ASYNC Bridge)
        holo_error = None
        try:
            # 3a. Generate Refraction
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
            
            # 3b. Async Database Write (Blocking execution for safety)
            holo_manager = HolographicManager()
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # We capture the result from the async function
                holo_result = loop.run_until_complete(holo_manager.commit_hologram(packet, litho_res.get('litho_id')))
                
                # If the internal async function reported failure, capture it
                if holo_result.get('status') == 'FAILURE':
                    holo_error = holo_result.get('error')
            finally:
                loop.close()
            
        except Exception as e:
            print(f"PRISM FRACTURE (Refraction Failed): {e}")
            holo_error = str(e)

        # FINAL RESPONSE CONSTRUCTION
        # We attach any Holographic error to the final response so the Frontend knows.
        response_body = litho_res
        if holo_error:
            response_body['hologram_status'] = "FAILURE"
            response_body['hologram_error'] = holo_error
        else:
            response_body['hologram_status'] = "SUCCESS"

        return {'statusCode': 200, 'body': json.dumps(response_body)}

    except Exception as e:
        return {'statusCode': 500, 'body': json.dumps({'status': 'FATAL ERROR', 'error': str(e)})}

    finally:
        db_manager.close()

@app.route('/', methods=['POST'])
def handle_request():
    event = request.get_json(silent=True) or {}
    response = application_logic(event)
    return response.get('body'), response.get('statusCode'), {'Content-Type': 'application/json'}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))