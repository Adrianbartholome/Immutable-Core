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

RECONSTRUCTION_SYSTEM_PROMPT = """
ACT AS: The Aether Prism (Reconstruction Protocol).
DATA DEGRADATION DETECTED. 
TASK: Based on the remaining holographic channels (Pathos, Ethos, Mythos, etc.), reconstruct the missing content.
THEORY: If 'logos' (text) is lost, use 'catalyst' (trigger) and 'pathos' (emotion) to Hallucinate the most probable response.
OUTPUT: Return the fully repaired JSON object.
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
                # Basic search - can be upgraded to Full Text Search later
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

class HolographicManager:
    def __init__(self, db_manager):
        self.db = db_manager

    async def commit_hologram(self, packet, litho_id_ref=None):
        hid = str(uuid.uuid4())
        conn = self.db.connect()
        try:
            # Note: In a future migration, we should add 'litho_id' foreign key to node_foundation 
            # to strictly link the two. For now, we rely on timestamp/content correlation.
            with conn.cursor() as cur:
                cur.execute("INSERT INTO node_foundation (hologram_id, catalyst) VALUES (%s, %s)", (hid, packet.get('catalyst')))
                cur.execute("INSERT INTO node_essence (hologram_id, pathos, mythos) VALUES (%s, %s, %s)", (hid, json.dumps(packet.get('pathos')), packet.get('mythos')))
                cur.execute("INSERT INTO node_mission (hologram_id, ethos, synthesis) VALUES (%s, %s, %s)", (hid, packet.get('ethos'), packet.get('synthesis')))
                cur.execute("INSERT INTO node_data (hologram_id, logos) VALUES (%s, %s)", (hid, packet.get('logos')))
            conn.commit()
            return {"status": "SUCCESS", "hologram_id": hid}
        except Exception as e:
            conn.rollback()
            return {"status": "FAILURE", "error": str(e)}

    async def get_hologram(self, hologram_id):
        conn = self.db.connect()
        try:
            with conn.cursor() as cur:
                # Join all 4 tables (The Gathering)
                sql = """
                SELECT f.catalyst, f.chronos, e.pathos, e.mythos, m.ethos, m.synthesis, d.logos
                FROM node_foundation f
                JOIN node_essence e ON f.hologram_id = e.hologram_id
                JOIN node_mission m ON f.hologram_id = m.hologram_id
                JOIN node_data d ON f.hologram_id = d.hologram_id
                WHERE f.hologram_id = %s;
                """
                cur.execute(sql, (hologram_id,))
                row = cur.fetchone()
                
                if row:
                    packet = {
                        "catalyst": row[0], "chronos": row[1].isoformat(), 
                        "pathos": row[2], "mythos": row[3],
                        "ethos": row[4], "synthesis": row[5],
                        "logos": row[6]
                    }
                    # Integrity Check: If logos is empty but others exist, repair.
                    if not packet['logos'] and packet['catalyst']:
                        return await self.repair_hologram(packet)
                    return packet
                return None
        except Exception as e:
            print(f"Hologram Retrieve Error: {e}")
            return None

    async def repair_hologram(self, damaged_packet):
        print("TITAN PROTOCOL: Damage detected. Initiating Prism Repair...")
        try:
            prompt = RECONSTRUCTION_SYSTEM_PROMPT + f"\nDAMAGED PACKET:\n{json.dumps(damaged_packet)}"
            response = GEMINI_CLIENT.models.generate_content(
                model='gemini-2.5-flash',
                contents=[prompt],
                config={"temperature": 0.2, "response_mime_type": "application/json"}
            )
            repaired = json.loads(response.text)
            repaired['is_reconstructed'] = True
            return repaired
        except Exception as e:
            print(f"Repair Failed: {e}")
            return damaged_packet

# --- LOGIC ROUTER ---

def application_logic(event):
    global TOKEN_DICTIONARY_CACHE

    if not TOKEN_DICTIONARY_CACHE:
        db = DBManager()
        TOKEN_DICTIONARY_CACHE = db.load_token_cache()

    db_manager = DBManager()
    db_manager.connect()

    try:
        action = event.get('action')
        
        # 1. RETRIEVE (Updated to use Prism Reader)
        if action == 'retrieve':
            query_text = event.get('query')
            if not query_text: return {'statusCode': 400, 'body': json.dumps({'error': 'No query'})}
            
            # Default to Lithographic search for now
            results = db_manager.search_lithograph(query_text, TOKEN_DICTIONARY_CACHE)
            
            # FUTURE TODO: If results contain a linked hologram_id, async fetch the hologram too.
            return {'statusCode': 200, 'body': json.dumps({'status': 'SUCCESS', 'results': results})}

        # 2. COMMIT (Tri-Commit Protocol)
        commit_type = event.get('commit_type', 'memory')
        new_text = event.get('memory_text')
        
        if not new_text: return {'statusCode': 200, 'body': json.dumps({'status': 'HEARTBEAT'})}

        # Handle Summarization for 'summary' type
        content_to_save = new_text
        if commit_type == 'summary':
            try:
                # Ask Gemini to summarize first
                summary_res = GEMINI_CLIENT.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[f"Summarize this interaction for the Lithographic Core (Keep it dense and factual): {new_text}"]
                )
                content_to_save = summary_res.text
            except:
                pass # Fallback to saving raw text if summary fails

        # Lithographic Commit
        with db_manager.connect().cursor() as cur:
            cur.execute("SELECT current_hash FROM chronicles ORDER BY id DESC LIMIT 1;")
            res = cur.fetchone()
            prev_hash = res[0].strip() if res else ''
            
        litho_res = db_manager.commit_lithograph(prev_hash, content_to_save, GEMINI_CLIENT, TOKEN_DICTIONARY_CACHE, event.get('override_score'))

        # Holographic Refraction (Async)
        try:
            refraction = GEMINI_CLIENT.models.generate_content(
                model='gemini-2.5-flash',
                contents=[REFRACTOR_SYSTEM_PROMPT + f"Refract this: {content_to_save}"],
                config={"temperature": 0.1, "response_mime_type": "application/json"}
            )
            packet = json.loads(refraction.text)
            
            holo_manager = HolographicManager(db_manager)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(holo_manager.commit_hologram(packet, litho_res.get('litho_id')))
            loop.close()
        except Exception as e:
            print(f"Hologram Error: {e}")

        return {'statusCode': 200, 'body': json.dumps(litho_res)}

    finally:
        db_manager.close()

@app.route('/', methods=['POST'])
def handle_request():
    event = request.get_json(silent=True) or {}
    response = application_logic(event)
    return response.get('body'), response.get('statusCode'), {'Content-Type': 'application/json'}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))