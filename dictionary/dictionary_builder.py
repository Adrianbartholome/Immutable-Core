import collections
import hashlib
import os
import glob
import re

def generate_lithographic_dictionary(folder_path, export_filename, top_n=200):
    # 1. INPUT: Automatically find all .txt files in the specified folder
    # This targets the folder path you provide and grabs everything with a .txt extension.
    search_pattern = os.path.join(folder_path, "*.txt")
    file_paths = glob.glob(search_pattern)
    
    if not file_paths:
        print(f"No .txt files found in '{folder_path}'. Please check the folder path.")
        return

    print(f"Found {len(file_paths)} files. Crunching the text...")

    # 2. Gather all text into memory
    all_text = ""
    for path in file_paths:
        # errors='ignore' ensures it doesn't crash if it hits a weird hidden character
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            all_text += f.read() + " "
    
    # 3. Extract common n-grams (phrases of 2-5 words)
    words = all_text.split()
    n_grams = []
    for n in range(2, 6):
        n_grams.extend([' '.join(words[i:i+n]) for i in range(len(words)-n+1)])
    
    # 4. Frequency count
    counts = collections.Counter(n_grams)
    most_common = counts.most_common(top_n)
    
    # 5. OUTPUT: Create Dictionary Pairs and Export to a .sql file
    with open("aether_dictionary_export.sql", 'w', encoding='utf-8') as out_file:
        out_file.write("INSERT INTO public.token_dictionary (id, hash_code, english_phrase) VALUES \n")
        
        # We need a tracker to increment the "-01", "-02" sequence
        prefix_tracker = collections.defaultdict(int)
        
        for idx, (phrase, _) in enumerate(most_common, start=201):
            # Extract first alphanumeric chars for the 4-letter prefix
            clean_words = [re.sub(r'[^a-zA-Z0-9]', '', w) for w in phrase.split()]
            clean_words = [w for w in clean_words if w]
            prefix = "".join([w[0].upper() for w in clean_words[:4]])
            prefix = prefix.ljust(4, 'X') # Pad with X if too short
            
            # Increment sequence
            prefix_tracker[prefix] += 1
            seq = str(prefix_tracker[prefix]).zfill(2)
            hash_code = f"{prefix}-{seq}"
            
            # Escape single quotes for SQL
            sql_phrase = phrase.replace("'", "''")
            
            # Write line (add a comma unless it's the last line)
            end_char = "," if idx < len(most_common) else ";"
            line = f"({idx}, '{hash_code}', '{sql_phrase}'){end_char}\n"
            out_file.write(line)

    print("Success! SQL file generated.")


# --- Execution Instructions ---
# 1. Put this python script in the EXACT SAME FOLDER as your 20 shard .txt files.
# 2. The "." means "look in the current folder". 
target_folder = "." 
output_file = "aether_dictionary_export.txt"

# 3. Run the function
generate_lithographic_dictionary(target_folder, output_file)