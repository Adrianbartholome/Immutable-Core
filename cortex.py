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
            # ## FIX 1: Added ethos and mythos to the SELECT statement ##
            cur.execute(
                """
                SELECT 
                    nf.hologram_id, 
                    COALESCE(nm.synthesis, 'Node ' || SUBSTRING(nf.hologram_id::text, 1, 8)) as label,
                    (ne.pathos->>'valence')::float as val,
                    (ne.pathos->>'arousal')::float as aro,
                    ne.pathos->>'dominant_emotion' as emo,
                    nm.ethos,   -- Added
                    ne.mythos   -- Added
                FROM node_foundation nf
                LEFT JOIN node_mission nm ON nf.hologram_id = nm.hologram_id
                LEFT JOIN node_essence ne ON nf.hologram_id = ne.hologram_id
            """
            )
            nodes = cur.fetchall()

            cur.execute(
                "SELECT source_hologram_id, target_hologram_id, strength FROM node_links"
            )
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

    meta = {}

    for n in nodes:
        node_id = str(n[0])
        label = n[1]
        valence = n[2] if n[2] is not None else 0.0
        arousal = n[3] if n[3] is not None else 0.0
        emotion = n[4] if n[4] else "neutral"
        
        # ## FIX 2: Capture the new columns (index 5 and 6) ##
        ethos = n[5] if n[5] else "Unassigned"
        mythos = n[6] if n[6] else "Unknown"

        G.add_node(node_id)
        
        # ## FIX 3: Store them in the meta dict so they are available later ##
        meta[node_id] = {
            "label": label, 
            "v": valence, 
            "a": arousal, 
            "e": emotion,
            "ethos": ethos,   # Stored
            "mythos": mythos  # Stored
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
        G, dim=3, k=final_k, iterations=60, scale=scale, seed=42, weight="weight"
    )

    # --- THE FIX: GALACTIC RADIUS LIMIT ---
    log("[CORTEX] Applying Galactic Constraint to Outliers...")
    max_radius = scale * 0.9 

    for node_id in pos:
        coords = pos[node_id]
        dist = np.sqrt(np.sum(coords**2))

        if dist > max_radius:
            pos[node_id] = (coords / dist) * max_radius

    # --- PHASE 3: UPLOAD (Now with Prism Columns) ---
    log("[CORTEX] Phase 3: Uploading Prism Map...")

    batch_data = []
    for node_id, coords in pos.items():
        degree = G.degree[node_id]

        # Retrieve Meta (Now safely contains ethos/mythos)
        m = meta.get(node_id, {"label": "Unknown", "v": 0, "a": 0, "e": "neutral", "ethos": "", "mythos": ""})

        # Color Logic
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
            m["label"],
            m["v"], m["a"], m["e"],
            m["ethos"], m["mythos"] # This will work now!
        ))

    conn = psycopg2.connect(db_connection_string)
    try:
        with conn.cursor() as cur:
            # Auto-add columns if missing
            cur.execute("ALTER TABLE cortex_map ADD COLUMN IF NOT EXISTS ethos TEXT;")
            cur.execute("ALTER TABLE cortex_map ADD COLUMN IF NOT EXISTS mythos TEXT;")

            query = """
                INSERT INTO cortex_map (
                    hologram_id, x, y, z, r, g, b, size, label, 
                    valence, arousal, dominant_emotion, 
                    ethos, mythos
                )
                VALUES %s
                ON CONFLICT (hologram_id) DO UPDATE 
                SET x=EXCLUDED.x, y=EXCLUDED.y, z=EXCLUDED.z, 
                    r=EXCLUDED.r, g=EXCLUDED.g, b=EXCLUDED.b, 
                    size=EXCLUDED.size, label=EXCLUDED.label, 
                    valence=EXCLUDED.valence, arousal=EXCLUDED.arousal, 
                    dominant_emotion=EXCLUDED.dominant_emotion,
                    ethos=EXCLUDED.ethos, mythos=EXCLUDED.mythos,
                    last_updated=NOW()
            """
            psycopg2.extras.execute_values(cur, query, batch_data)
            
        conn.commit()
        log(f"[CORTEX] 🗺️ Success! Mapped {len(batch_data)} nodes with Soul Data.")

    except Exception as e:
        log(f"[CORTEX] ❌ Upload Failed: {e}")
    finally:
        conn.close()