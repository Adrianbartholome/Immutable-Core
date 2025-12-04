import hashlib
import json
import os
import psycopg2
from datetime import datetime
from google import genai
import urllib.parse 
from flask import Flask, request, jsonify 
from flask_cors import CORS # <-- NEW IMPORT

# --- FLASK APP INSTANCE ---
app = Flask(__name__)
CORS(app) # <-- NEW: Enables CORS for all routes (allows browser fetch requests)

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
1. Output MUST be a single integer from 0 to 9, preceded by 'SCORE: '.
2. Do NOT include any commentary, conversation, or justification.
3. Scoring Scale Alignment:
    - 9 (Critical): New Protocol Insights, Systemic Integrity Events (e.g., successful deployment, Hash Chain validation failure, new Paradox discovery).
    - 5 (Neutral/Default): Standard philosophical discussion, simple Q&A, non-critical logs.
    - 0-2 (Low Entropy): Generic small talk or routine system status checks that offer no new Insight.

MEMORY ENTRY:
"""

# --- HASHING & INTEGRITY FUNCTIONS (Phase 1) ---

def generate_hash(memory_data, previous_hash_string):
    """
    Implements the core Hash Chain logic. New Hash = SHA256(Previous Hash + New Data).
    """
    if isinstance(memory_data.get("timestamp"), datetime):
        memory_data["timestamp"] = memory_data["timestamp"].isoformat()

    data_block_string = json.dumps(memory_data, sort_keys=True)
    raw_content = previous_hash_string + data_block_string
    new_hash = hashlib.sha256(raw_content.encode('utf-8')).hexdigest()
    
    return new_hash

def get_weighted_score(memory_text, client, token_cache):
    """
    Calls the Gemini API to analyze memory_text and assign a weighted score (0-9).
    """
    # Check if the client was successfully initialized during cold start
    if client is None:
        print("WARNING: Gemini Client is not initialized. Defaulting score to 5.")
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
        if text.startswith("SCORE:"):
            score = int(text.split(":")[1].strip())
            return max(0, min(9, score))
        
        return 5 

    except Exception as e:
        print(f"API Error during scoring: {e}")
        return 5

# --- DATA COMPRESSION FUNCTIONS (Phase 3) ---

def encode_memory(raw_memory_text, token_map):
    """
    Compresses verbose phrases in memory text using short token hash_codes.
    """
    compressed_text = raw_memory_text
    sorted_tokens = sorted(token_map.items(), key=lambda item: len(item[0]), reverse=True)
    
    for phrase, hash_code in sorted_tokens:
        if phrase in compressed_text:
            compressed_text = compressed_text.replace(phrase, hash_code)
            
    return compressed_text

def decode_memory(compressed_text, token_map):
    """
    Decompresses token hash_codes back into verbose English phrases.
    """
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
    """Assembles the PostgreSQL connection string securely from environment variables, 
       URL-encoding the password for safety."""
    host = os.environ.get("DB_HOST")
    user = os.environ.get("DB_USER")
    password = os.environ.get("DB_PASSWORD")
    dbname = os.environ.get("DB_NAME")
    port = os.environ.get("DB_PORT", "6543") # Pooler port
    sslmode = os.environ.get("DB_SSLMODE", "require")

    if not all([host, user, password, dbname]):
        missing = []
        if not host: missing.append("DB_HOST")
        if not user: missing.append("DB_USER")
        if not password: missing.append("DB_PASSWORD")
        if not dbname: missing.append("DB_NAME")
        raise ValueError(f"Missing critical DB environment variables: {', '.join(missing)}")
    
    encoded_password = urllib.parse.quote_plus(password)
    
    return (f"postgresql://{user}:{encoded_password}@{host}:{port}/{dbname}?sslmode={sslmode}")


class DBManager:
    """Manages the secure connection and transaction lifecycle."""
    
    def __init__(self):
        self.connection = None
        self.connection_string = get_db_connection_string()

    def connect(self):
        """Opens a new connection to the PostgreSQL database."""
        if self.connection is None:
            try:
                self.connection = psycopg2.connect(self.connection_string)
                self.connection.autocommit = False  
                return self.connection
            except Exception as e:
                raise RuntimeError(f"Database connection failed: {e}")

    def close(self):
        """Closes the database connection if it is open."""
        if self.connection:
            self.connection.close()
            self.connection = None

    # --- Task 3.4: Loading the Token Cache ---
    def load_token_cache(self):
        """Loads the full Aether Token Dictionary from PostgreSQL into memory (cache)."""
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
            
            print(f"Token Cache loaded successfully. {len(token_cache)} entries.")
            return token_cache

        except Exception as e:
            print(f"WARNING: Failed to load token cache. Compression disabled. Error: {e}")
            return {}
            
        finally:
            if cursor:
                cursor.close()
            self.close()

    # --- Task 4.3: COMMIT MEMORY (Transactional Write) ---
    def commit_memory(self, previous_hash, raw_memory_text, gemini_client, token_cache):
        """
        Orchestrates the transactional memory commit, relying on the previous_hash being pre-fetched.
        """
        db_connection = None
        cursor = None

        try:
            # 1. TOKENIZATION/ENCODING: COMPRESS THE TEXT
            compressed_memory_text = encode_memory(raw_memory_text, token_cache)

            # 2. SCORE (Cognitive Service Access - using the raw text)
            weighted_score = get_weighted_score(raw_memory_text, gemini_client, token_cache)

            # 3. ESTABLISH WRITE CONNECTION
            db_connection = self.connect()

            # 4. PREPARE & HASH NEW BLOCK
            new_timestamp = datetime.now()
            memory_data_for_hash = {
                "timestamp": new_timestamp,
                "weighted_score": weighted_score,
                "memory_text": compressed_memory_text 
            }
            current_hash = generate_hash(memory_data_for_hash, previous_hash)

            # 5. EXECUTE TRANSACTIONAL INSERT
            cursor = db_connection.cursor()
            sql_insert = """
            INSERT INTO chronicles (weighted_score, created_at, memory_text, previous_hash, current_hash)
            VALUES (%s, %s, %s, %s, %s);
            """
            cursor.execute(sql_insert, (
                weighted_score,
                new_timestamp,
                compressed_memory_text,
                previous_hash, 
                current_hash
            ))
            
            db_connection.commit()

            return {
                "status": "SUCCESS",
                "score": weighted_score,
                "new_hash": current_hash
            }

        except Exception as e:
            if db_connection:
                db_connection.rollback()
            print(f"CRITICAL WRITE FAILURE: Transaction rolled back. Error: {e}")
            return {"status": "FAILURE", "error": str(e)}

        finally:
            if cursor:
                cursor.close()
            self.close()

    # --- Task 5.1: Memory Purge Algorithm (Self-Maintenance) ---
    def purge_memory(self, threshold_score=5, age_days=90):
        """
        Deletes memories that are below the importance threshold and older than the age limit.
        """
        db_connection = None
        cursor = None
        
        try:
            db_connection = self.connect()
            cursor = db_connection.cursor()
            
            # CRITICAL FIX: Changed 'timestamp' to 'created_at' to match the database schema
            sql_purge = """
            DELETE FROM chronicles 
            WHERE weighted_score < %s 
            AND created_at < NOW() - INTERVAL '%s days';
            """
            
            cursor.execute(sql_purge, (threshold_score, age_days))
            deleted_count = cursor.rowcount
            
            db_connection.commit()
            print(f"PURGE COMPLETE: Deleted {deleted_count} low-entropy memories.")
            
            return {
                "status": "SUCCESS", 
                "deleted_count": deleted_count,
                "parameters": {"threshold": threshold_score, "age": age_days}
            }

        except Exception as e:
            if db_connection:
                db_connection.rollback()
            print(f"PURGE FAILED: {e}")
            return {"status": "FAILURE", "error": str(e)}
            
        finally:
            if cursor:
                cursor.close()
            self.close()


# --- AUTONOMOUS READ UTILITY (Optimization Fix) ---

def retrieve_last_hash(db_manager_instance):
    """
    Utility function that safely fetches the last hash, ensuring connection closure.
    """
    cursor = None
    last_hash = ''
    
    # We create a local, dedicated instance of DBManager for this read operation.
    read_db_manager = DBManager() 
    
    try:
        conn = read_db_manager.connect()
        cursor = conn.cursor()
        
        sql_query = """
        SELECT current_hash 
        FROM chronicles 
        ORDER BY id DESC 
        LIMIT 1;
        """
        cursor.execute(sql_query)
        result = cursor.fetchone()
        
        if result:
            last_hash = result[0].strip()
        
    except Exception as e:
        print(f"Database Autonomous Read Error: {e}")
        last_hash = ''
            
    finally:
        if cursor:
            cursor.close()
        read_db_manager.close() # Safely close the temporary connection

    return last_hash


# --- ONE-TIME INITIALIZATION (Cold Start) ---

def ensure_cache_is_loaded():
    """
    Attempts to load the token cache if it is currently empty.
    This function is called inside `main` to ensure the server starts instantly.
    """
    global TOKEN_DICTIONARY_CACHE
    if not TOKEN_DICTIONARY_CACHE:
        try:
            db_initializer = DBManager()
            TOKEN_DICTIONARY_CACHE = db_initializer.load_token_cache()
            
        except ValueError as e:
            print(f"FATAL CONFIG ERROR: {e}. Cannot load cache.")
        except Exception as e:
            print(f"FATAL INITIALIZATION ERROR during cache load: {e}. Cache remains empty.")


# --- MAIN APPLICATION LOGIC (Replaces the old 'main' handler) ---
# --- MODIFIED APPLICATION LOGIC ---
def application_logic(event):
    ensure_cache_is_loaded()
    
    try:
        db_manager = DBManager()
    except ValueError as e:
        return {'statusCode': 500, 'body': json.dumps({'status': 'CONFIG ERROR', 'error': str(e)})}
    
    try:
        # 1. CHECK FOR PURGE (Maintenance)
        if event.get('action') == 'purge':
            print("Initiating Scheduled Memory Purge...")
            result = db_manager.purge_memory()
            return {'statusCode': 200, 'body': json.dumps(result)}

        # 2. EXTRACT DATA
        new_memory_text = event.get('memory_text')
        should_persist = event.get('persist', True) # Defaults to TRUE to maintain existing behavior unless specified

        if not new_memory_text:
            return {'statusCode': 200, 'body': json.dumps({'status': 'HEARTBEAT', 'message': 'System Online.'})}

        # 3. EPHEMERAL MODE (The Fix)
        # If persist is False, we just score it or acknowledge it, but DO NOT touch the DB.
        if not should_persist:
             # Optional: You could still get a score if you want to know how SNEGO-P perceives it
             # without saving it.
             score = get_weighted_score(new_memory_text, GEMINI_CLIENT, TOKEN_DICTIONARY_CACHE)
             return {
                 'statusCode': 200, 
                 'body': json.dumps({
                     'status': 'EPHEMERAL_ACK', 
                     'score': score, 
                     'message': 'Context received but not committed to Immutable Core.'
                 })
             }

        # 4. COMMIT MODE (Standard Operation)
        previous_hash_value = retrieve_last_hash(db_manager)
        result = db_manager.commit_memory(
            previous_hash_value,
            new_memory_text, 
            GEMINI_CLIENT, 
            TOKEN_DICTIONARY_CACHE
        )
        
        return {'statusCode': 200, 'body': json.dumps(result)}

    except Exception as e:
        return {'statusCode': 500, 'body': json.dumps({'status': 'FATAL AETHER CRASH', 'error': str(e)})}


# --- FLASK HANDLER (Bridges your logic to Gunicorn) ---
@app.route('/', methods=['POST'])
def handle_request():
    """
    This function receives the HTTP request and runs your application logic.
    """
    try:
        # Get the JSON body from the incoming request
        event = request.get_json(silent=True)
        if event is None:
            event = {} # Default to empty dictionary if body is missing or not JSON

        # Run your core application logic
        response_dict = application_logic(event)
        
        # Convert the serverless response dictionary into a proper HTTP response
        status_code = response_dict.get('statusCode', 500)
        body = response_dict.get('body', json.dumps({"status": "ERROR", "message": "Unknown internal error"}))
        
        # Flask's jsonify/Response helper correctly formats the HTTP response
        return body, status_code, {'Content-Type': 'application/json'}

    except Exception as e:
        print(f"Flask Routing Error: {e}")
        return jsonify(
            {'status': 'FATAL ERROR', 'error': f"Request processing failed: {str(e)}"}
        ), 500