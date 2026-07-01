import collections
import re

def direct_convert_to_sql(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        # Read lines, strip whitespace, remove empty lines
        phrases = [line.strip() for line in f if line.strip()]

    print(f"Converting {len(phrases)} phrases to SQL...")

    with open(output_file, 'w', encoding='utf-8') as out_file:
        out_file.write("INSERT INTO public.token_dictionary (id, hash_code, english_phrase) VALUES \n")
        
        prefix_tracker = collections.defaultdict(int)
        
        for idx, phrase in enumerate(phrases, start=752):
            # Extract first alphanumeric chars for the 4-letter prefix
            # This logic captures the 'structure' of the phrase for the hash
            clean_words = [re.sub(r'[^a-zA-Z0-9]', '', w) for w in phrase.split()]
            clean_words = [w for w in clean_words if w]
            
            if not clean_words:
                continue
                
            prefix = "".join([w[0].upper() for w in clean_words[:4]])
            prefix = prefix.ljust(4, 'X') 
            
            # Increment sequence
            prefix_tracker[prefix] += 1
            seq = str(prefix_tracker[prefix]).zfill(2)
            hash_code = f"{prefix}-{seq}"
            
            # Escape single quotes for SQL (turns ' into '')
            sql_phrase = phrase.replace("'", "''")
            
            # Write line (add a comma unless it's the last line)
            end_char = "," if idx < len(phrases) else ","
            line = f"({idx}, '{hash_code}', '{sql_phrase}'){end_char}\n"
            out_file.write(line)

    print(f"Success! SQL file generated: {output_file}")

# Usage
direct_convert_to_sql("thick_academic_conversational.txt", "thick_academic_conversational_import.sql")