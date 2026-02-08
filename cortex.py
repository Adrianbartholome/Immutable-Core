import networkx as nx
import numpy as np
import psycopg2
import psycopg2.extras
import time

# --- CONFIGURATION ---
# Adjust these to tune the "look" of the brain
SCALE = 1000.0  # How spread out the nodes are
ITERATIONS = 50 # Higher = more stable, slower
DIMENSIONS = 3  # 3D
GRAVITY = 0.1   # Pulls nodes to center

def regenerate_neural_map(db_connection_string):
    print("[CORTEX] 🗺️  Starting Neural Cartography...")
    start_time = time.time()
    
    # --- STEP 1: FETCH DATA (Connect -> Fetch -> Disconnect) ---
    print("[CORTEX] Phase 1: Downloading Structure...")
    nodes = []
    edges = []
    
    conn = psycopg2.connect(db_connection_string)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT hologram_id FROM node_foundation")
            nodes = cur.fetchall()
            
            cur.execute("SELECT source_hologram_id, target_hologram_id, strength FROM node_links")
            edges = cur.fetchall()
    finally:
        conn.close() # <--- IMPORTANT: We hang up here!

    # --- STEP 2: CALCULATE PHYSICS (Offline - Take as long as you want) ---
    print(f"[CORTEX] Phase 2: Calculating Forces for {len(nodes)} nodes...")
    
    G = nx.Graph()
    for n in nodes:
        G.add_node(str(n[0]))
        
    for source, target, strength in edges:
        s_str, t_str = str(source), str(target)
        if s_str in G and t_str in G:
            weight = float(strength) if strength else 1.0
            G.add_edge(s_str, t_str, weight=weight)

    node_count = G.number_of_nodes()
    if node_count == 0:
        print("[CORTEX] ⚠️  Empty Graph. Skipping.")
        return

    # This is the heavy part (45s+)
    k_val = 1.0 / np.sqrt(node_count) if node_count > 0 else None
    pos = nx.spring_layout(G, dim=DIMENSIONS, k=k_val, iterations=ITERATIONS, scale=SCALE, seed=42)

    # Prepare data for upload
    batch_data = []
    for node_id, coords in pos.items():
        degree = G.degree[node_id]
        r, g, b = 100, 200, 255 
        size = 1.0
        
        if degree > 5: 
            size = 2.0
            r, g, b = 255, 100, 255
        if degree > 10:
            size = 4.0
            r, g, b = 255, 200, 50

        batch_data.append((node_id, float(coords[0]), float(coords[1]), float(coords[2]), r, g, b, size))

    # --- STEP 3: UPLOAD DATA (Connect -> Save -> Disconnect) ---
    print("[CORTEX] Phase 3: Uploading Map...")
    
    conn = psycopg2.connect(db_connection_string) # <--- Re-dialing the server
    try:
        with conn.cursor() as cur:
            # Ensure table exists
            cur.execute("""
                CREATE TABLE IF NOT EXISTS cortex_map (
                    hologram_id UUID PRIMARY KEY,
                    x FLOAT, y FLOAT, z FLOAT,
                    r INTEGER, g INTEGER, b INTEGER, size FLOAT,
                    last_updated TIMESTAMP DEFAULT NOW()
                );
            """)
            
            query = """
                INSERT INTO cortex_map (hologram_id, x, y, z, r, g, b, size)
                VALUES %s
                ON CONFLICT (hologram_id) DO UPDATE 
                SET x=EXCLUDED.x, 
                    y=EXCLUDED.y, 
                    z=EXCLUDED.z, 
                    r=EXCLUDED.r, 
                    g=EXCLUDED.g, 
                    b=EXCLUDED.b, 
                    size=EXCLUDED.size, 
                    last_updated=NOW()
            """
            psycopg2.extras.execute_values(cur, query, batch_data)
            
        conn.commit()
        duration = time.time() - start_time
        print(f"[CORTEX] 🗺️  Success! Mapped {len(batch_data)} nodes in {duration:.2f}s.")

    except Exception as e:
        print(f"[CORTEX] ❌ Upload Failed: {e}")
    finally:
        conn.close()

# Usage:
# regenerate_neural_map("postgresql://user:pass@host/dbname")