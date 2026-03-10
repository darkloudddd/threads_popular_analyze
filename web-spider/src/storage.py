import json
import os

def save_results(data, path="result.jsonl", append=False):
    """
    Saves data to a file. 
    If data is a list of strings, it saves as plain text (legacy).
    If data is a list of dicts, it saves as JSONL.
    """
    if not data:
        return

    mode = "a" if append else "w"
    is_jsonl = path.endswith(".jsonl") or (isinstance(data[0], dict))
    
    with open(path, mode, encoding="utf-8") as f:
        for item in data:
            if isinstance(item, dict):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            else:
                # Legacy fallback for plain text
                f.write(str(item).replace("\n", " ") + "\n")
