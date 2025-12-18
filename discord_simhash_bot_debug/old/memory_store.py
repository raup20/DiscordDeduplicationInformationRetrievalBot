import json
from discord_simhash_bot_debug.old.simhash_engine import simhash, hamming_distance

DB_FILE = "messages.json"

def load_db():
    try:
        with open(DB_FILE, "r") as f:
            return json.load(f)
    except:
        return []

def save_db(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=4)

def store_message(msg):
    db = load_db()
    db.append({"text": msg, "hash": simhash(msg)})
    save_db(db)

def find_similar(message, threshold=30):
    db = load_db()
    query_hash = simhash(message)
    results = []
    for entry in db:
        dist = hamming_distance(query_hash, entry["hash"])
        results.append((entry["text"], dist))
    results.sort(key=lambda x: x[1])
    return [(m,d) for m,d in results if d <= threshold]
