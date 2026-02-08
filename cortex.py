import networkx as nx
import numpy as np
import psycopg2
import psycopg2.extras
import time

# Default Physics
DEFAULT_SCALE = 1000.0
DEFAULT_ITERATIONS = 50 
DIMENSIONS = 3

def regenerate_neural_map(db_connection_string, spacing=1.0, cluster_strength=1.0, status_callback=None):
    """
    Mission-Aware Cartographer: Maps the 'synthesis' from node_mission to the star.
    """
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    log(f"[CORTEX] 🗺️  Starting Cartography (Spacing: {spacing}, Cluster: {cluster_strength})...")
    start_time = time.time()
    
    conn = psycopg2.connect(db_connection_string)
    
    # --- PHASE 1: FETCH DATA ---
    log("[CORTEX] Phase 1: Downloading Structure & Synthesis...")
    nodes = []
    edges = []
    
    try:
        with conn.cursor() as cur:
            # JOIN node_foundation with node_mission
            # Uses 'synthesis' as the label. If null, falls back to ID.
            cur.execute("""
                SELECT 
                    nf.hologram_id, 
                    COALESCE(nm.synthesis, 'Node ' || SUBSTRING(nf.hologram_id::text, 1, 8)) as label
                FROM node_foundation nf
                LEFT JOIN node_mission nm ON nf.hologram_id = nm.hologram_id
            """)
            nodes = cur.fetchall()
            
            # Fetch Synapses
            cur.execute("SELECT source_hologram_id, target_hologram_id, strength FROM node_links")
            edges = cur.fetchall()
            
    except Exception as e:
        log(f"[CORTEX] ⚠️ SQL Error (Check node_mission table!): {e}")
        return
    finally:
        conn.close()

    if not nodes:
        log("[CORTEX] ⚠️  Empty Graph. Skipping.")
        return

    # --- PHASE 2: CALCULATE PHYSICS ---
    log(f"[CORTEX] Phase 2: Calculating Forces for {len(nodes)} nodes...")
    G = nx.Graph()
    
    labels = {}
    for n in nodes:
        node_id = str(n[0])
        label = n[1] 
        G.add_node(node_id)
        labels[node_id] = label
        
    for source, target, strength in edges:
        s_str, t_str = str(source), str(target)
        if s_str in G and t_str in G:
            base_weight = float(strength) if strength else 1.0
            G.add_edge(s_str, t_str, weight=base_weight * cluster_strength)

    # Physics
    node_count = G.number_of_nodes()
    base_k = 1.0 / np.sqrt(node_count) if node_count > 0 else 0.1
    final_k = base_k * spacing

    pos = nx.spring_layout(G, dim=3, k=final_k, iterations=50, scale=1000.0, seed=42)

    # --- PHASE 3: UPLOAD ---
    log("[CORTEX] Phase 3: Uploading Map...")
    
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
            # Ensure tables exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS cortex_map (
                    hologram_id UUID PRIMARY KEY,
                    x FLOAT, y FLOAT, z FLOAT,
                    r INTEGER, g INTEGER, b INTEGER, size FLOAT,
                    label TEXT,
                    last_updated TIMESTAMP DEFAULT NOW()
                );
            """)
            cur.execute("ALTER TABLE cortex_map ADD COLUMN IF NOT EXISTS label TEXT;")

            # Upsert
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
        log(f"[CORTEX] 🗺️  Success! Mapped {len(batch_data)} nodes in {time.time() - start_time:.2f}s")
    except Exception as e:
        log(f"[CORTEX] ❌ Upload Failed: {e}")
    finally:
        conn.close()