import numpy as np
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text: str) -> np.ndarray:
    v = _model.encode([text], normalize_embeddings=True)[0]
    return v.astype(np.float32)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    # if embeddings are normalized, cosine = dot
    return float(np.dot(a, b))
