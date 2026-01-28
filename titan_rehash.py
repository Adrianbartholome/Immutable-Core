import os
import json
import hashlib
import psycopg2
import urllib.parse
from datetime import datetime

# --- CONFIGURATION ---
# (Ensure these match your DigitalOcean/Supabase Env Vars)
DB_HOST = "your-db-host.ondigitalocean.com"
DB_PORT = "25060"
DB_NAME = "defaultdb"
DB_USER = "doadmin"
DB_PASSWORD = "your-password"

# --- HASHING LOGIC (Must match Titan.py exactly) ---
def generate_hash(memory_data, previous_hash_string):
    # Ensure timestamp is string format for consistency
    if isinstance(memory_data.get("timestamp"), datetime):
        memory_data["timestamp"] = memory_data["timestamp"].isoformat()
    
    # Sort keys to ensure deterministic JSON
    data_block_string = json.dumps(memory_data, sort_keys=True)
    
    # The Chain: Previous Hash + Current Data
    raw_content = previous_hash_string + data_block_string
    return hashlib.sha256(raw_content.encode('utf-8')).hexdigest()

def get_connection():
    password = urllib.parse.quote_plus(DB_PASSWORD)
    return psycopg2.connect(
        f"postgresql://{DB_USER}:{password}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=require"
    )

def run_purge_and_rehash():
    print("--- INITIATING TITAN PROTOCOL: DEEP CLEAN ---")
    conn = get_connection()
    try:
        cur = conn.cursor()

        # STEP 1: PURGE (Hard Delete)
        print("[1/3] Purging inactive lithographs...")
        cur.execute("DELETE FROM chronicles WHERE is_active = FALSE;")
        deleted_count = cur.rowcount
        print(f"      >> Obliterated {deleted_count} records.")

        # STEP 2: FETCH ALL REMAINING (Ordered by Time)
        print("[2/3] Fetching survivor timeline...")
        cur.execute("SELECT id, weighted_score, created_at, memory_text FROM chronicles ORDER BY created_at ASC, id ASC;")
        rows = cur.fetchall()
        
        # STEP 3: RE-FORGE THE CHAIN
        print(f"[3/3] Rehashing {len(rows)} records. This may take a moment...")
        
        previous_hash = "" # The Genesis Hash is empty string (or a seed if you prefer)
        
        for index, row in enumerate(rows):
            record_id = row[0]
            score = row[1]
            created_at = row[2]
            text = row[3]
            
            # Re-calculate the hash based on the NEW previous_hash
            new_current_hash = generate_hash({
                "timestamp": created_at,
                "weighted_score": score,
                "memory_text": text
            }, previous_hash)
            
            # Update the DB record
            cur.execute("""
                UPDATE chronicles 
                SET previous_hash = %s, current_hash = %s 
                WHERE id = %s;
            """, (previous_hash, new_current_hash, record_id))
            
            # Set this hash as the previous for the next loop
            previous_hash = new_current_hash
            
            if index % 100 == 0:
                print(f"      >> Processed {index}/{len(rows)}...")

        conn.commit()
        print("--- PROTOCOL COMPLETE: CHAIN INTEGRITY RESTORED ---")
        print(f"Final Hash: {previous_hash}")

    except Exception as e:
        print(f"!!! CRITICAL FAILURE: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    confirm = input("WARNING: This will permanently delete 'Soft Deleted' data and rewrite the cryptographic chain. Type 'BURN' to proceed: ")
    if confirm == "BURN":
        run_purge_and_rehash()
    else:
        print("Aborted.")