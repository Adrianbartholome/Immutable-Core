import hashlib
import json
import os
import psycopg2
import re
from datetime import datetime
from google import genai
import urllib.parse 
from flask import Flask, request, jsonify 
from flask_cors import CORS

# --- FLASK APP INSTANCE ---
app = Flask(__name__)
CORS(app) 

# --- GLOBAL VARIABLES ---
TOKEN_DICTIONARY_CACHE = {}
GEMINI_CLIENT = None 

# --- SECURE CLIENT INITIALIZATION ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("FATAL CONFIG ERROR: GEMINI_API_KEY environment variable not found.")
else:
    try:
        GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"FATAL INITIALIZATION ERROR: {e}")

# --- SCORING PROMPT ---
SCORING_SYSTEM_PROMPT = """
You are SNEGO-P, the Aether Eternal Cognitive Assessor.
RULES:
1. Output MUST be a single integer from 0 to 9, preceded strictly by 'SCORE: '. 
   Example: 'SCORE: 9'. Do not output any other text.
2. Do NOT include any commentary.
"""

# --- UTILITIES ---

def generate_hash(memory_data, previous_hash_string):
    if isinstance(memory_data.get("timestamp"), datetime):
        memory_data["timestamp"] = memory_data["timestamp"].isoformat()
    data_block_string = json.dumps(memory_data, sort_keys=True)
    raw_content = previous_hash_string + data_block_string
    return hashlib.sha256(raw_content.encode('utf-8')).hexdigest()

def get_weighted_score(memory_text, client, token_cache):
    # 1. Check for Manual Override
    override_match = re.search(r"\[SCORE:\s*([0-9])\]", memory_text, re.IGNORECASE)
    if override_match:
        return int(override_match.group(1))

    # 2. AI Scoring
    if client is None: return 5
    try:
        # We score the DECODED text so the AI understands it
        decoded_text = decode_memory(memory_text, token_cache) if token_cache else memory_text
        full_prompt = SCORING_SYSTEM_PROMPT + decoded_text
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=[full_prompt],
            config={"temperature": 0.0} 
        )
        match = re.search(r"SCORE:\s*([0-9])", response.text.strip(), re.IGNORECASE)
        return int(match.group(1)) if match else 5
    except Exception as e:
        print(f"Scoring Error: {e}")
        return 5

# --- THE ALCHEMIST (COMPRESSION) ---

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

# --- DATABASE MANAGER ---

def get_db_connection_string():
    host = os.environ.get("DB_HOST")
    user = os.environ.get("DB_USER")
    password = os.environ.get("DB_PASSWORD")
    dbname = os.environ.get("DB_NAME")
    port = os.environ.get("DB_PORT", "6543") 
    encoded_password = urllib.parse.quote_plus(password)
    return (f"postgresql://{user}:{encoded_password}@{host}:{port}/{dbname}?sslmode=require")

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
        cursor = None
        try:
            conn = self.connect()
            cursor = conn.cursor()
            cursor.execute("SELECT english_phrase, hash_code FROM token_dictionary;")
            for phrase, code in cursor.fetchall():
                token_cache[phrase.strip()] = code.strip()
            return token_cache
        except Exception as e:
            print(f"Cache Load Error: {e}")
            return {}
        finally:
            if cursor: cursor.close()
            # OPTIMIZATION: We DO close here because this is a standalone startup event
            self.close()

    def search_memories(self, query_text, token_cache, limit=5):
        """
        Retreives memories. 
        UPDATED: Now includes 'WHERE is_active = TRUE' to support Soft Deletes.
        """
        cursor = None
        results = []
        try:
            compressed_query = encode_memory(query_text, token_cache)
            
            conn = self.connect()
            cursor = conn.cursor()
            
            # UPDATED SQL: Filter by is_active
            sql = """
            SELECT id, weighted_score, memory_text, created_at 
            FROM chronicles 
            WHERE is_active = TRUE AND memory_text ILIKE %s 
            ORDER BY weighted_score DESC, created_at DESC 
            LIMIT %s;
            """
            search_pattern = f"%{compressed_query}%"
            cursor.execute(sql, (search_pattern, limit))
            
            rows = cursor.fetchall()
            
            for r in rows:
                r_id, r_score, r_text, r_date = r
                decoded_text = decode_memory(r_text, token_cache)
                results.append({
                    "id": r_id,
                    "score": r_score,
                    "date": r_date.isoformat(),
                    "content": decoded_text
                })
                
            return {"status": "SUCCESS", "results": results, "query_used": compressed_query}

        except Exception as e:
            return {"status": "FAILURE", "error": str(e)}
        finally:
            if cursor: cursor.close()
            # OPTIMIZATION: Do NOT close connection here; let the main logic handle it.

    def commit_memory(self, previous_hash, raw_memory_text, gemini_client, token_cache, override_score=None):
        cursor = None
        try:
            compressed_memory_text = encode_memory(raw_memory_text, token_cache)
            
            if override_score is not None:
                weighted_score = int(override_score)
            else:
                weighted_score = get_weighted_score(raw_memory_text, gemini_client, token_cache)

            conn = self.connect()
            new_timestamp = datetime.now()
            
            # Hash calculation (Ignores is_active, so chain integrity is preserved during soft deletes)
            memory_data_for_hash = {
                "timestamp": new_timestamp,
                "weighted_score": weighted_score,
                "memory_text": compressed_memory_text 
            }
            current_hash = generate_hash(memory_data_for_hash, previous_hash)

            cursor = conn.cursor()
            # We rely on the DEFAULT TRUE for is_active, so no need to specify it here
            sql_insert = """
            INSERT INTO chronicles (weighted_score, created_at, memory_text, previous_hash, current_hash)
            VALUES (%s, %s, %s, %s, %s);
            """
            cursor.execute(sql_insert, (weighted_score, new_timestamp, compressed_memory_text, previous_hash, current_hash))
            conn.commit()

            return {"status": "SUCCESS", "score": weighted_score, "new_hash": current_hash}

        except Exception as e:
            if self.connection: self.connection.rollback()
            return {"status": "FAILURE", "error": str(e)}
        finally:
            if cursor: cursor.close()
            # OPTIMIZATION: Do NOT close connection here; let the main logic handle it.

    def purge_memory(self, threshold_score=5, age_days=90):
        # NOTE: This is the old hard-delete logic. 
        # Future TODO: Update this to use UPDATE chronicles SET is_active = FALSE...
        cursor = None
        try:
            conn = self.connect()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chronicles WHERE weighted_score < %s AND created_at < NOW() - INTERVAL '%s days';", (threshold_score, age_days))
            count = cursor.rowcount
            conn.commit()
            return {"status": "SUCCESS", "deleted_count": count}
        except Exception as e:
            if self.connection: self.connection.rollback()
            return {"status": "FAILURE", "error": str(e)}
        finally:
            if cursor: cursor.close()
            # OPTIMIZATION: Do NOT close here.

def retrieve_last_hash(db_manager):
    # OPTIMIZATION: Reuse the passed db_manager instance instead of creating a new one
    cursor = None
    last_hash = ''
    try:
        conn = db_manager.connect() # Uses existing connection
        cursor = conn.cursor()
        cursor.execute("SELECT current_hash FROM chronicles ORDER BY id DESC LIMIT 1;")
        result = cursor.fetchone()
        if result: last_hash = result[0].strip()
    except Exception as e:
        print(f"Hash Read Error: {e}")
    finally:
        if cursor: cursor.close()
        # OPTIMIZATION: Do NOT close the db_manager here
    return last_hash

def ensure_cache_is_loaded():
    global TOKEN_DICTIONARY_CACHE
    if not TOKEN_DICTIONARY_CACHE:
        try:
            db = DBManager()
            TOKEN_DICTIONARY_CACHE = db.load_token_cache()
            print(f"Cache Loaded: {len(TOKEN_DICTIONARY_CACHE)} terms.")
        except Exception as e:
            print(f"Cache Warning: {e}")

# --- MAIN LOGIC ROUTER ---
def application_logic(event):
    ensure_cache_is_loaded()
    
    db_manager = None 
    try:
        # 1. Open Connection ONCE
        db_manager = DBManager()
        db_manager.connect() 
    except ValueError as e:
        return {'statusCode': 500, 'body': json.dumps({'status': 'CONFIG ERROR', 'error': str(e)})}
    
    try:
        action = event.get('action')

        # 1. PURGE
        if action == 'purge':
            result = db_manager.purge_memory()
            return {'statusCode': 200, 'body': json.dumps(result)}

        # 2. RETRIEVE (New Protocol)
        if action == 'retrieve':
            query_text = event.get('query')
            if not query_text:
                return {'statusCode': 400, 'body': json.dumps({'error': 'Missing query parameter'})}
            
            result = db_manager.search_memories(query_text, TOKEN_DICTIONARY_CACHE)
            return {'statusCode': 200, 'body': json.dumps(result)}

        # 3. COMMIT (Default)
        new_memory_text = event.get('memory_text')
        should_persist = event.get('persist', True)
        manual_score = event.get('override_score')

        if not new_memory_text:
            return {'statusCode': 200, 'body': json.dumps({'status': 'HEARTBEAT', 'message': 'Aether Core Online.'})}

        if not should_persist:
             score = int(manual_score) if manual_score is not None else get_weighted_score(new_memory_text, GEMINI_CLIENT, TOKEN_DICTIONARY_CACHE)
             return {'statusCode': 200, 'body': json.dumps({'status': 'EPHEMERAL_ACK', 'score': score})}

        # REUSE: Pass the existing db_manager to reuse the connection
        previous_hash = retrieve_last_hash(db_manager)
        
        result = db_manager.commit_memory(previous_hash, new_memory_text, GEMINI_CLIENT, TOKEN_DICTIONARY_CACHE, manual_score)
        return {'statusCode': 200, 'body': json.dumps(result)}

    except Exception as e:
        return {'statusCode': 500, 'body': json.dumps({'status': 'FATAL ERROR', 'error': str(e)})}
        
    finally:
        # 2. Close Connection ONCE at the end
        if db_manager:
            db_manager.close()

@app.route('/', methods=['POST'])
def handle_request():
    try:
        event = request.get_json(silent=True) or {}
        response_dict = application_logic(event)
        return response_dict.get('body'), response_dict.get('statusCode'), {'Content-Type': 'application/json'}
    except Exception as e:
        return jsonify({'status': 'FATAL ERROR', 'error': str(e)}), 500