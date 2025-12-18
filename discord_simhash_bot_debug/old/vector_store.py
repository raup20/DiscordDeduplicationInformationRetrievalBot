import json
import numpy as np
from embed_engine import embed, cosine_sim

DB_FILE = "vec_messages.json"

def load_db():
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # store vectors as numpy arrays
        for e in raw:
            e["vec"] = np.array(e["vec"], dtype=np.float32)
        return raw
    except:
        return []

def save_db(db):
    out = []
    for e in db:
        out.append({"text": e["text"], "vec": e["vec"].tolist()})
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

def store_message(text: str):
    db = load_db()
    db.append({"text": text, "vec": embed(text)})
    save_db(db)

def find_similar(text: str, top_k=3, min_sim=0.65):
    db = load_db()
    q = embed(text)
    scored = []
    for e in db:
        s = cosine_sim(q, e["vec"])
        scored.append((e["text"], s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(t, s) for (t, s) in scored[:top_k] if s >= min_sim]
