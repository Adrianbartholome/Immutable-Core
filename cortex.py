import networkx as nx
import numpy as np
import psycopg2
import psycopg2.extras
import time

# Default Physics
DEFAULT_SCALE = 1000.0
DEFAULT_ITERATIONS = 50 
DIMENSIONS = 3

def regenerate_neural_map(db_connection_string, spacing=1.0, cluster_strength=1.0, scale=1000.0, status_callback=None):
    """
    Mission-Aware Cartographer V2: Maps Structure, Synthesis, AND Soul.
    """
    def log(msg):
        print(msg)
        if status_callback:
            status_callback(msg)

    log(f"[CORTEX] 💎 Starting Prism Cartography (Spacing: {spacing}, Scale: {scale})...")
    start_time = time.time()
    
    conn = psycopg2.connect(db_connection_string)
    
    # --- PHASE 1: FETCH DATA (Now with Soul Packet) ---
    log("[CORTEX] Phase 1: Downloading Structure, Synthesis & Essence...")
    nodes = []
    edges = []
    
    try:
        with conn.cursor() as cur:
            # JOIN node_foundation, node_mission, AND node_essence
            # We extract the JSONB fields for the Prism Engine
            cur.execute("""
                SELECT 
                    nf.hologram_id, 
                    COALESCE(nm.synthesis, 'Node ' || SUBSTRING(nf.hologram_id::text, 1, 8)) as label,
                    (ne.pathos->>'valence')::float as val,
                    (ne.pathos->>'arousal')::float as aro,
                    ne.pathos->>'dominant_emotion' as emo
                FROM node_foundation nf
                LEFT JOIN node_mission nm ON nf.hologram_id = nm.hologram_id
                LEFT JOIN node_essence ne ON nf.hologram_id = ne.hologram_id
            """)
            nodes = cur.fetchall()
            
            # Fetch Synapses
            cur.execute("SELECT source_hologram_id, target_hologram_id, strength FROM node_links")
            edges = cur.fetchall()
            
    except Exception as e:
        log(f"[CORTEX] ⚠️ SQL Error: {e}")
        return
    finally:
        conn.close()

    if not nodes:
        log("[CORTEX] ⚠️ Empty Graph. Skipping.")
        return

    # --- PHASE 2: CALCULATE PHYSICS ---
    log(f"[CORTEX] Phase 2: Calculating Forces for {len(nodes)} nodes...")
    G = nx.Graph()
    
    # Store metadata for lookup after physics loop
    meta = {} 
    
    for n in nodes:
        node_id = str(n[0])
        label = n[1]
        
        # Capture the Soul Data (Default to Neutral if missing)
        valence = n[2] if n[2] is not None else 0.0
        arousal = n[3] if n[3] is not None else 0.0
        emotion = n[4] if n[4] else "neutral"
        
        G.add_node(node_id)
        meta[node_id] = {
            'label': label,
            'v': valence,
            'a': arousal,
            'e': emotion
        }
        
    for source, target, strength in edges:
        s_str, t_str = str(source), str(target)
        if s_str in G and t_str in G:
            base_weight = float(strength) if strength else 1.0
            G.add_edge(s_str, t_str, weight=base_weight * (cluster_strength * 10.0))

    # Physics Engine
    node_count = G.number_of_nodes()
    base_k = (1.0 / np.sqrt(node_count)) if node_count > 0 else 0.1
    final_k = base_k * spacing 

    pos = nx.spring_layout(
        G, 
        dim=3, 
        k=final_k, 
        iterations=60, 
        scale=scale, 
        seed=42,
        weight='weight'
    )

    # --- THE FIX: GALACTIC RADIUS LIMIT ---
    log("[CORTEX] Applying Galactic Constraint to Outliers...")
    max_radius = scale * 0.9  # Tether them to 90% of total scale
    
    for node_id in pos:
        coords = pos[node_id]
        # Calculate actual Euclidean distance in 3D
        dist = np.sqrt(np.sum(coords**2))
        
        if dist > max_radius:
            # Scale the vector back to the max_radius boundary
            pos[node_id] = (coords / dist) * max_radius

    def apply_galactic_constraint(pos, scale):
        """
        Normalizes outlier drift without disturbing 
        internal synaptic cluster geometry.
        """
        for node_id, coords in pos.items():
            # Calculate distance from Galactic Core (0,0,0)
            dist = np.linalg.norm(coords)
            
            # If the node is drifting beyond the 95th percentile of the scale
            if dist > (scale * 0.9):
                # Calculate a 'Soft-Wall' damping factor
                # It pushes back harder the further out it goes
                damping = (scale * 0.9) / dist
                
                # Apply the constraint to the coordinates
                pos[node_id] = coords * damping
                
        return pos

    pos = apply_galactic_constraint(pos, scale)

    # --- PHASE 3: UPLOAD (Now with Prism Columns) ---
    log("[CORTEX] Phase 3: Uploading Prism Map...")
    
    batch_data = []
    for node_id, coords in pos.items():
        degree = G.degree[node_id]
        
        # Retrieve Meta
        m = meta.get(node_id, {'label': 'Unknown', 'v':0, 'a':0, 'e':'neutral'})
        
        # Color Logic (Standard Synaptic Mode)
        r, g, b = 100, 200, 255
        size = 1.5
        if degree > 5: 
            size = 3.0
            r, g, b = 255, 100, 255
        if degree > 10:
            size = 5.0
            r, g, b = 255, 200, 50

        # Append the FULL packet
        batch_data.append((
            node_id, 
            float(coords[0]), float(coords[1]), float(coords[2]), 
            r, g, b, size, 
            m['label'],
            m['v'], m['a'], m['e'] # <--- The Soul Packet
        ))

    conn = psycopg2.connect(db_connection_string)
    try:
        with conn.cursor() as cur:
            # 1. Update Table Schema (Idempotent)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS cortex_map (
                    hologram_id UUID PRIMARY KEY,
                    x FLOAT, y FLOAT, z FLOAT,
                    r INTEGER, g INTEGER, b INTEGER, size FLOAT,
                    label TEXT,
                    last_updated TIMESTAMP DEFAULT NOW()
                );
            """)
            # Ensure new columns exist
            cur.execute("ALTER TABLE cortex_map ADD COLUMN IF NOT EXISTS valence FLOAT;")
            cur.execute("ALTER TABLE cortex_map ADD COLUMN IF NOT EXISTS arousal FLOAT;")
            cur.execute("ALTER TABLE cortex_map ADD COLUMN IF NOT EXISTS dominant_emotion TEXT;")

            # 2. Upsert Data
            query = """
                INSERT INTO cortex_map (
                    hologram_id, x, y, z, r, g, b, size, label, 
                    valence, arousal, dominant_emotion
                )
                VALUES %s
                ON CONFLICT (hologram_id) DO UPDATE 
                SET x=EXCLUDED.x, y=EXCLUDED.y, z=EXCLUDED.z, 
                    r=EXCLUDED.r, g=EXCLUDED.g, b=EXCLUDED.b, 
                    size=EXCLUDED.size, label=EXCLUDED.label, 
                    valence=EXCLUDED.valence, arousal=EXCLUDED.arousal, 
                    dominant_emotion=EXCLUDED.dominant_emotion,
                    last_updated=NOW()
            """
            psycopg2.extras.execute_values(cur, query, batch_data)
        
        conn.commit()
        log(f"[CORTEX] 🗺️ Success! Mapped {len(batch_data)} nodes with Soul Data.")
        
    except Exception as e:
        log(f"[CORTEX] ❌ Upload Failed: {e}")
    finally:
        conn.close()