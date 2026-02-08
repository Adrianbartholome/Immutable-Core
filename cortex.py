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
    """
    Reads the Logical Graph -> Computes Physics -> Writes to Visual Graph.
    """
    print("[CORTEX] 🗺️  Starting Neural Cartography...")
    start_time = time.time()
    
    conn = psycopg2.connect(db_connection_string)
    
    try:
        # --- 1. BUILD THE GRAPH ---
        G = nx.Graph()
        
        with conn.cursor() as cur:
            # Fetch ALL Nodes (We need them all for the physics to work right)
            print("[CORTEX] Fetching Nodes...")
            cur.execute("SELECT hologram_id FROM node_foundation")
            nodes = cur.fetchall()
            for n in nodes:
                G.add_node(str(n[0]))
                
            # Fetch Edges (Synapses)
            print("[CORTEX] Fetching Synapses...")
            cur.execute("SELECT source_hologram_id, target_hologram_id, strength FROM node_links")
            edges = cur.fetchall()
            for source, target, strength in edges:
                s_str, t_str = str(source), str(target)
                if s_str in G and t_str in G:
                    # Physics Trick: High strength = Short spring (tight cluster)
                    # We invert strength for 'weight' because spring_layout treats weight as attraction
                    weight = float(strength) if strength else 1.0
                    G.add_edge(s_str, t_str, weight=weight)

        node_count = G.number_of_nodes()
        if node_count == 0:
            print("[CORTEX] ⚠️  Empty Graph. Skipping.")
            return

        print(f"[CORTEX] Graph Built: {node_count} Nodes | {G.number_of_edges()} Synapses")

        # --- 2. COMPUTE PHYSICS (The Heavy Math) ---
        print("[CORTEX] Calculating Forces (3D Spring Layout)...")
        # k = Optimal distance between nodes. 
        # For large graphs, k = 1/sqrt(n) is a good starting point.
        k_val = 1.0 / np.sqrt(node_count) if node_count > 0 else None
        
        pos = nx.spring_layout(
            G, 
            dim=DIMENSIONS, 
            k=k_val, 
            iterations=ITERATIONS, 
            scale=SCALE, 
            seed=42 # Fixed seed keeps the brain from "jumping" every time you re-map
        )

        # --- 3. SAVE TO VISUAL LAYER ---
        print("[CORTEX] Anchoring Coordinates to 'cortex_map'...")
        
        # Prepare batch data
        batch_data = []
        for node_id, coords in pos.items():
            # Basic coloring logic (You can make this smarter later based on tags)
            # Default: White/Cyan
            r, g, b = 100, 200, 255 
            size = 1.0
            
            # If node has many connections, make it bigger/brighter
            degree = G.degree[node_id]
            if degree > 5: 
                size = 2.0
                r, g, b = 255, 100, 255 # (Pink/Purple)
            if degree > 10:
                size = 4.0
                r, g, b = 255, 200, 50 # (Gold)

            batch_data.append((
                node_id, 
                float(coords[0]), float(coords[1]), float(coords[2]), 
                r, g, b, 
                size
            ))

        with conn.cursor() as cur:
            # Upsert (Insert or Update if exists)
            query = """
                INSERT INTO cortex_map (hologram_id, x, y, z, r, g, b, size, last_updated)
                VALUES %s
                ON CONFLICT (hologram_id) DO UPDATE 
                SET x = EXCLUDED.x, 
                    y = EXCLUDED.y, 
                    z = EXCLUDED.z,
                    r = EXCLUDED.r,
                    g = EXCLUDED.g,
                    b = EXCLUDED.b,
                    size = EXCLUDED.size,
                    last_updated = NOW();
            """
            psycopg2.extras.execute_values(cur, query, batch_data)
            
        conn.commit()
        duration = time.time() - start_time
        print(f"[CORTEX] 🗺️  Mapping Complete in {duration:.2f}s.")

    except Exception as e:
        print(f"[CORTEX] ❌ Mapping Failed: {e}")
    finally:
        conn.close()

# Usage:
# regenerate_neural_map("postgresql://user:pass@host/dbname")