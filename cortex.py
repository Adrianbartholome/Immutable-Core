import networkx as nx
import numpy as np
import psycopg2
import psycopg2.extras
import time

# Default Physics
DEFAULT_SCALE = 1000.0
DEFAULT_ITERATIONS = 50 
DIMENSIONS = 3

def regenerate_neural_map(db_connection_string, spacing=1.0, cluster_strength=1.0):
    """
    spacing: Controls 'k' (Optimal distance). Higher = more spread out islands.
    cluster_strength: Multiplier for edge weights. Higher = tighter local groups.
    """
    print(f"[CORTEX] 🗺️  Starting Cartography (Spacing: {spacing}, Cluster: {cluster_strength})...")
    start_time = time.time()
    
    conn = psycopg2.connect(db_connection_string)
    
    # --- PHASE 1: FETCH DATA & TEXT ---
    print("[CORTEX] Downloading Structure & Memories...")
    try:
        with conn.cursor() as cur:
            # Fetch Node IDs AND a snippet of text
            # We assume node_foundation links to lithographic_ledger via lithograph_id
            # Adjust the JOIN if your schema is different!
            cur.execute("""
                SELECT nf.hologram_id, LEFT(ll.memory_text, 100) as label
                FROM node_foundation nf
                LEFT JOIN lithographic_ledger ll ON nf.lithograph_id = ll.lithograph_id
            """)
            nodes = cur.fetchall()
            
            cur.execute("SELECT source_hologram_id, target_hologram_id, strength FROM node_links")
            edges = cur.fetchall()
    finally:
        conn.close()

    if not nodes:
        print("[CORTEX] ⚠️  Empty Graph.")
        return

    # --- PHASE 2: CALCULATE PHYSICS ---
    print(f"[CORTEX] Calculating Forces for {len(nodes)} nodes...")
    G = nx.Graph()
    
    # Map ID -> Label for later
    labels = {}
    for n in nodes:
        node_id = str(n[0])
        label = n[1] if n[1] else "Unknown Artifact"
        G.add_node(node_id)
        labels[node_id] = label
        
    for source, target, strength in edges:
        s_str, t_str = str(source), str(target)
        if s_str in G and t_str in G:
            # CLUMPING LOGIC: Multiply strength by user's cluster_strength
            # High cluster_strength = stronger springs = tighter clumps
            base_weight = float(strength) if strength else 1.0
            G.add_edge(s_str, t_str, weight=base_weight * cluster_strength)

    # SPACING LOGIC: 'k' determines global spacing.
    # Standard is 1/sqrt(n). We multiply by user 'spacing' factor.
    # Spacing > 1.0 = Islands push apart. Spacing < 1.0 = Big blob.
    node_count = G.number_of_nodes()
    base_k = 1.0 / np.sqrt(node_count) if node_count > 0 else 0.1
    final_k = base_k * spacing

    pos = nx.spring_layout(G, dim=DIMENSIONS, k=final_k, iterations=DEFAULT_ITERATIONS, scale=DEFAULT_SCALE, seed=42)

    # --- PHASE 3: UPLOAD ---
    print("[CORTEX] Uploading Map...")
    batch_data = []
    for node_id, coords in pos.items():
        degree = G.degree[node_id]
        label = labels.get(node_id, "Unknown")
        
        # Color Logic
        r, g, b = 100, 200, 255
        size = 1.5
        if degree > 5: 
            size = 3.0
            r, g, b = 255, 100, 255
        if degree > 10:
            size = 5.0
            r, g, b = 255, 200, 50

        batch_data.append((node_id, float(coords[0]), float(coords[1]), float(coords[2]), r, g, b, size, label))

    conn = psycopg2.connect(db_connection_string)
    try:
        with conn.cursor() as cur:
            # Add 'label' column if missing
            cur.execute("""
                CREATE TABLE IF NOT EXISTS cortex_map (
                    hologram_id UUID PRIMARY KEY,
                    x FLOAT, y FLOAT, z FLOAT,
                    r INTEGER, g INTEGER, b INTEGER, size FLOAT,
                    label TEXT,
                    last_updated TIMESTAMP DEFAULT NOW()
                );
            """)
            
            # Use this trick to add the column safely if table already exists
            cur.execute("ALTER TABLE cortex_map ADD COLUMN IF NOT EXISTS label TEXT;")

            query = """
                INSERT INTO cortex_map (hologram_id, x, y, z, r, g, b, size, label)
                VALUES %s
                ON CONFLICT (hologram_id) DO UPDATE 
                SET x=EXCLUDED.x, y=EXCLUDED.y, z=EXCLUDED.z, 
                    r=EXCLUDED.r, g=EXCLUDED.g, b=EXCLUDED.b, 
                    size=EXCLUDED.size, label=EXCLUDED.label, 
                    last_updated=NOW()
            """
            psycopg2.extras.execute_values(cur, query, batch_data)
        conn.commit()
        print(f"[CORTEX] 🗺️  Done in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"[CORTEX] ❌ Upload Failed: {e}")
    finally:
        conn.close()