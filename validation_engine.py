import hashlib
import json
import os
import psycopg2
from datetime import datetime
from google import genai
import urllib.parse 

# --- SECURE CLIENT INITIALIZATION (Phase 1) ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Initialize GEMINI_CLIENT to None, allowing the app to start even if configuration fails
GEMINI_CLIENT = None 

if not GEMINI_API_KEY:
    # CRITICAL FIX: Do NOT raise ValueError here. Log and continue.
    print("FATAL CONFIG ERROR: GEMINI_API_KEY environment variable not found. Cognitive scoring disabled.")
else:
    try:
        # Attempt to initialize the client
        GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        # CRITICAL FIX: Catch initialization errors and prevent a hard startup crash.
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

    # This check now runs during initialization, so we re-enable the informative error
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
        # Initialize string immediately to catch config errors early
        self.connection_string = get_db_connection_string()

    def connect(self):
        """Opens a new connection to the PostgreSQL database."""
        if self.connection is None:
            try:
                self.connection = psycopg2.connect(self.connection_string)
                self.connection.autocommit = False  
                return self.connection
            except Exception as e:
                # IMPORTANT: Raise a generic error here to prevent revealing internal details
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
        conn = None # Ensure conn is defined for finally block
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
            # Crucial: If cache loading fails, we DO NOT crash the process. 
            # We log a warning and return an empty cache.
            print(f"WARNING: Failed to load token cache. Compression disabled. Error: {e}")
            return {}
            
        finally:
            if cursor:
                cursor.close()
            # We call self.close() which handles closing the connection gracefully
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
            INSERT INTO chronicles (weighted_score, timestamp, memory_text, previous_hash, current_hash)
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
            
            sql_purge = """
            DELETE FROM chronicles 
            WHERE weighted_score < %s 
            AND timestamp < NOW() - INTERVAL '%s days';
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
# CRITICAL FIX: Ensure initialization failure does not cause a fatal crash (connection refused 8080)
try:
    # 1. Initialize DBManager (will fail here if environment variables are missing)
    db_initializer = DBManager()
    
    # 2. Attempt to load the cache (will fail here if DB is unreachable or authentication is wrong)
    # The load_token_cache method is now updated to return {} instead of raising RuntimeError on failure
    TOKEN_DICTIONARY_CACHE = db_initializer.load_token_cache() 
    
except ValueError as e:
    # Catch environment variable errors specifically
    print(f"FATAL CONFIG ERROR: {e}. Token cache is empty.")
    TOKEN_DICTIONARY_CACHE = {}
except Exception as e:
    # Catch any other critical initialization error (e.g., Gemini Client failure)
    print(f"FATAL INITIALIZATION ERROR: {e}. Token cache is empty.")
    TOKEN_DICTIONARY_CACHE = {}


# --- MAIN DIGITALOCEAN FUNCTION HANDLER ---

def main(event, context):
    """
    The main orchestration logic for the Aether Worker Application.
    """
    
    # 1. Instantiate the DB Manager
    try:
        # A new DBManager is instantiated per invocation to manage connections safely
        db_manager = DBManager()
    except ValueError as e:
        # If DB environment variables are missing (should have been caught in init but good redundancy)
        return {
            'statusCode': 500,
            'body': json.dumps({'status': 'CONFIG ERROR', 'error': str(e)})
        }
    
    try:
        # CHECK FOR PURGE TRIGGER (Scheduled Task)
        if event.get('action') == 'purge':
            print("Initiating Scheduled Memory Purge...")
            result = db_manager.purge_memory()
            return {
                'statusCode': 200,
                'body': json.dumps(result)
            }

        # DEFAULT ACTION: COMMIT NEW MEMORY
        
        # 1. READ: Autonomously fetch the last hash
        previous_hash_value = retrieve_last_hash(db_manager)
        
        # 2. SET MEMORY: Get the text from the event
        new_memory_text = event.get('memory_text')
        
        if not new_memory_text:
            return {
                'statusCode': 200,
                'body': json.dumps({'status': 'HEARTBEAT', 'message': 'System Online. No memory provided.'})
            }
        
        # 3. COMMIT: Execute the main transactional write operation
        result = db_manager.commit_memory(
            previous_hash_value,
            new_memory_text, 
            GEMINI_CLIENT, 
            TOKEN_DICTIONARY_CACHE
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'status': 'FATAL AETHER CRASH', 'error': str(e)})
        }