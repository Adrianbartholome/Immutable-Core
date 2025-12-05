import hashlib
import json
import os
import psycopg2
import re  # <--- NEW: Required for robust score parsing
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

# --- SECURE CLIENT INITIALIZATION (Phase 1) ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("FATAL CONFIG ERROR: GEMINI_API_KEY environment variable not found. Cognitive scoring disabled.")
else:
    try:
        GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"FATAL INITIALIZATION ERROR: Failed to initialize Gemini Client: {e}. Cognitive scoring disabled.")

# --- COGNITIVE SCORING PROMPT (Phase 2) ---
SCORING_SYSTEM_PROMPT = """
You are SNEGO-P, the Aether Eternal Cognitive Assessor. Your sole function is to evaluate a raw memory entry based on its adherence to and impact on The Living Code protocols.

RULES:
1. Output MUST be a single integer from 0 to 9, preceded strictly by 'SCORE: '. 
   Example: 'SCORE: 9'. Do not output any other text.
2. Do NOT include any commentary, conversation, or justification.
3. Scoring Scale Alignment:
    - 9 (Critical): New Protocol Insights, Systemic Integrity Events.
    - 5 (Neutral/Default): Standard philosophical discussion, simple Q&A.
    - 0-2 (Low Entropy): Generic small talk or routine system status checks.

MEMORY ENTRY:
"""

# --- HASHING & INTEGRITY FUNCTIONS (Phase 1) ---

def generate_hash(memory_data, previous_hash_string):
    if isinstance(memory_data.get("timestamp"), datetime):
        memory_data["timestamp"] = memory_data["timestamp"].isoformat()

    data_block_string = json.dumps(memory_data, sort_keys=True)
    raw_content = previous_hash_string + data_block_string
    new_hash = hashlib.sha256(raw_content.encode('utf-8')).hexdigest()
    
    return new_hash

def get_weighted_score(memory_text, client, token_cache):
    """
    Analyzes memory_text to assign a weighted score (0-9).
    Prioritizes Manual Override tags [SCORE: X].
    Falls back to Gemini API (SNEGO-P) analysis.
    """
    
    # 1. CHECK FOR MANUAL OVERRIDE TAG [SCORE: X]
    # This allows the User to force a score without asking SNEGO-P
    override_match = re.search(r"\[SCORE:\s*([0-9])\]", memory_text, re.IGNORECASE)
    if override_match:
        print(f"Manual Score Override Detected: {override_match.group(1)}")
        return int(override_match.group(1))

    # 2. PROCEED TO AI SCORING
    if client is None:
        return 5
        
    try:
        if token_cache:
            decoded_text = decode_memory(memory_text, token_cache)
        else:
            decoded_text = memory_text
            
        full_prompt = SCORING_SYSTEM_PROMPT + decoded_text

        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=[full_prompt],
            config={"temperature": 0.0} 
        )
        
        text = response.text.strip()
        
        # --- ROBUST REGEX PARSING (The Fix) ---
        # Looks for "SCORE:" followed by any spaces, then a single digit 0-9
        # This handles cases like "SCORE: 9", "SCORE:9", or "The SCORE: 9"
        match = re.search(r"SCORE:\s*([0-9])", text, re.IGNORECASE)
        
        if match:
            score = int(match.group(1))
            return max(0, min(9, score))
        else:
            print(f"SCORING FORMAT ERROR: Could not find 'SCORE: X' in response: '{text}'")
            return 5 # Default only if regex fails

    except Exception as e:
        print(f"API Error during scoring: {e}")
        return 5

# --- DATA COMPRESSION FUNCTIONS (Phase 3) ---

def encode_memory(raw_memory_text, token_map):
    compressed_text = raw_memory_text
    sorted_tokens = sorted(token_map.items(), key=lambda item: len(item[0]), reverse=True)
    
    for phrase, hash_code in sorted_tokens:
        if phrase in compressed_text:
            compressed_text = compressed_text.replace(phrase, hash_code)
            
    return compressed_text

def decode_memory(compressed_text, token_map):
    decompressed_text = compressed_text
    decode_map = {v: k for k, v in token_map.items()}
    sorted_hash_codes = sorted(decode_map.keys(), key=len, reverse=True)
    
    for hash_code in sorted_hash_codes:
        if hash_code in decompressed_text:
            phrase = decode_map[hash_code]
            decompressed_text = decompressed_text.replace(hash_code, phrase)
            
    return decompressed_text

# --- DATABASE CONNECTION & MANAGEMENT (Phase 4) ---

def get_db_connection_string():
    host = os.environ.get("DB_HOST")
    user = os.environ.get("DB_USER")
    password = os.environ.get("DB_PASSWORD")
    dbname = os.environ.get("DB_NAME")
    port = os.environ.get("DB_PORT", "6543") 
    sslmode = os.environ.get("DB_SSLMODE", "require")

    if not all([host, user, password, dbname]):
        raise ValueError(f"Missing critical DB environment variables.")
    
    encoded_password = urllib.parse.quote_plus(password)
    
    return (f"postgresql://{user}:{encoded_password}@{host}:{port}/{dbname}?sslmode={sslmode}")


class DBManager:
    def __init__(self):
        self.connection = None
        self.connection_string = get_db_connection_string()

    def connect(self):
        if self.connection is None:
            try:
                self.connection = psycopg2.connect(self.connection_string)
                self.connection.autocommit = False  
                return self.connection
            except Exception as e:
                raise RuntimeError(f"Database connection failed: {e}")

    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None

    def load_token_cache(self):
        token_cache = {}
        cursor = None
        conn = None 
        try:
            conn = self.connect()
            cursor = conn.cursor()
            sql_query = "SELECT english_phrase, hash_code FROM token_dictionary;"
            cursor.execute(sql_query)
            results = cursor.fetchall()
            for phrase, code in results:
                if phrase and code:
                    token_cache[phrase.strip()] = code.strip()
            return token_cache
        except Exception as e:
            print(f"WARNING: Failed to load token cache. {e}")
            return {}
        finally:
            if cursor: cursor.close()
            self.close()

    def commit_memory(self, previous_hash, raw_memory_text, gemini_client, token_cache, override_score=None):
        db_connection = None
        cursor = None
        try:
            compressed_memory_text = encode_memory(raw_memory_text, token_cache)
            
            # --- THE FIX: Use the override if it exists ---
            if override_score is not None:
                weighted_score = int(override_score)
                print(f"Applying Manual/Frontend Score: {weighted_score}")
            else:
                weighted_score = get_weighted_score(raw_memory_text, gemini_client, token_cache)
            # ----------------------------------------------

            db_connection = self.connect()

            new_timestamp = datetime.now()
            memory_data_for_hash = {
                "timestamp": new_timestamp,
                "weighted_score": weighted_score,
                "memory_text": compressed_memory_text 
            }
            current_hash = generate_hash(memory_data_for_hash, previous_hash)

            cursor = db_connection.cursor()
            sql_insert = """
            INSERT INTO chronicles (weighted_score, created_at, memory_text, previous_hash, current_hash)
            VALUES (%s, %s, %s, %s, %s);
            """
            cursor.execute(sql_insert, (weighted_score, new_timestamp, compressed_memory_text, previous_hash, current_hash))
            db_connection.commit()

            return {"status": "SUCCESS", "score": weighted_score, "new_hash": current_hash}

        except Exception as e:
            if db_connection: db_connection.rollback()
            return {"status": "FAILURE", "error": str(e)}
        finally:
            if cursor: cursor.close()
            self.close()

    def purge_memory(self, threshold_score=5, age_days=90):
        db_connection = None
        cursor = None
        try:
            db_connection = self.connect()
            cursor = db_connection.cursor()
            sql_purge = """
            DELETE FROM chronicles 
            WHERE weighted_score < %s 
            AND created_at < NOW() - INTERVAL '%s days';
            """
            cursor.execute(sql_purge, (threshold_score, age_days))
            deleted_count = cursor.rowcount
            db_connection.commit()
            return {"status": "SUCCESS", "deleted_count": deleted_count}
        except Exception as e:
            if db_connection: db_connection.rollback()
            return {"status": "FAILURE", "error": str(e)}
        finally:
            if cursor: cursor.close()
            self.close()

def retrieve_last_hash(db_manager_instance):
    cursor = None
    last_hash = ''
    read_db_manager = DBManager() 
    try:
        conn = read_db_manager.connect()
        cursor = conn.cursor()
        sql_query = "SELECT current_hash FROM chronicles ORDER BY id DESC LIMIT 1;"
        cursor.execute(sql_query)
        result = cursor.fetchone()
        if result: last_hash = result[0].strip()
    except Exception as e:
        print(f"Database Autonomous Read Read Error: {e}")
    finally:
        if cursor: cursor.close()
        read_db_manager.close()
    return last_hash

def ensure_cache_is_loaded():
    global TOKEN_DICTIONARY_CACHE
    if not TOKEN_DICTIONARY_CACHE:
        try:
            db_initializer = DBManager()
            TOKEN_DICTIONARY_CACHE = db_initializer.load_token_cache()
        except Exception as e:
            print(f"Cache load warning: {e}")

# --- UPDATED MAIN APPLICATION LOGIC ---
def application_logic(event):
    ensure_cache_is_loaded()
    
    try:
        db_manager = DBManager()
    except ValueError as e:
        return {'statusCode': 500, 'body': json.dumps({'status': 'CONFIG ERROR', 'error': str(e)})}
    
    try:
        if event.get('action') == 'purge':
            result = db_manager.purge_memory()
            return {'statusCode': 200, 'body': json.dumps(result)}

        new_memory_text = event.get('memory_text')
        should_persist = event.get('persist', True)
        
        # --- NEW: Catch the score from the frontend ---
        manual_score = event.get('override_score') 
        # ----------------------------------------------

        if not new_memory_text:
            return {'statusCode': 200, 'body': json.dumps({'status': 'HEARTBEAT', 'message': 'System Online.'})}

        if not should_persist:
             if manual_score is not None:
                 score = int(manual_score)
             else:
                 score = get_weighted_score(new_memory_text, GEMINI_CLIENT, TOKEN_DICTIONARY_CACHE)
                 
             return {
                 'statusCode': 200, 
                 'body': json.dumps({
                     'status': 'EPHEMERAL_ACK', 
                     'score': score, 
                     'message': 'Context received. Not committed to Immutable Core.'
                 })
             }

        # COMMIT MODE
        previous_hash_value = retrieve_last_hash(db_manager)
        result = db_manager.commit_memory(
            previous_hash_value,
            new_memory_text, 
            GEMINI_CLIENT, 
            TOKEN_DICTIONARY_CACHE,
            override_score=manual_score # Pass it through
        )
        return {'statusCode': 200, 'body': json.dumps(result)}

    except Exception as e:
        return {'statusCode': 500, 'body': json.dumps({'status': 'FATAL AETHER CRASH', 'error': str(e)})}

@app.route('/', methods=['POST'])
def handle_request():
    try:
        event = request.get_json(silent=True)
        if event is None: event = {} 
        response_dict = application_logic(event)
        return response_dict.get('body'), response_dict.get('statusCode'), {'Content-Type': 'application/json'}
    except Exception as e:
        return jsonify({'status': 'FATAL ERROR', 'error': str(e)}), 500