import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from embed_engine import embed, cosine_sim
from srp_index import SRPIndex

Q_DB = "questions.json"
A_DB = "answers.json"

@dataclass
class QItem:
    msg_id: int
    channel_id: int
    author_id: int
    text: str
    ts: float
    vec: np.ndarray

@dataclass
class AItem:
    msg_id: int
    channel_id: int
    author_id: int
    text: str
    ts: float
    vec: np.ndarray
    qid: int  # linked question msg_id

class QAStore:
    """
    Stores Questions + Answers separately.

    Retrieval side (teacher suggestion):
      - Use embeddings (SentenceTransformer) + cosine similarity
      - Use an approximate-nearest-neighbor (ANN) *index* in embedding space
      - Then re-rank candidates with exact cosine similarity

    Practical note:
      - hnswlib/HNSW is great but often fails to build on Windows + Python 3.13.
      - This project uses a pure-Python Signed Random Projection (SRP) ANN index
        (random hyperplane signatures + banding) to avoid build issues.
    """
    def __init__(self, dim: int = 384, n_planes: int = 64, n_bands: int = 8):
        self.dim = dim

        self.questions: List[QItem] = []
        self.answers: List[AItem] = []
        self.answers_by_qid: Dict[int, List[AItem]] = {}

        # Fast lookup
        self.q_by_id: Dict[int, QItem] = {}

        # Recent questions per channel (for linking answers)
        self.recent_q_by_channel: Dict[int, List[int]] = {}

        # ANN index for questions (embedding space)
        self.q_index = SRPIndex(dim=self.dim, n_planes=n_planes, n_bands=n_bands)

        self._load_all()

    # ---------- persistence ----------
    def _load_questions(self) -> List[QItem]:
        try:
            raw = json.load(open(Q_DB, "r", encoding="utf-8"))
        except:
            return []
        out: List[QItem] = []
        for e in raw:
            out.append(
                QItem(
                    msg_id=int(e["msg_id"]),
                    channel_id=int(e["channel_id"]),
                    author_id=int(e["author_id"]),
                    text=e["text"],
                    ts=float(e["ts"]),
                    vec=np.array(e["vec"], dtype=np.float32),
                )
            )
        return out

    def _load_answers(self) -> List[AItem]:
        try:
            raw = json.load(open(A_DB, "r", encoding="utf-8"))
        except:
            return []
        out: List[AItem] = []
        for e in raw:
            out.append(
                AItem(
                    msg_id=int(e["msg_id"]),
                    channel_id=int(e["channel_id"]),
                    author_id=int(e["author_id"]),
                    text=e["text"],
                    ts=float(e["ts"]),
                    vec=np.array(e["vec"], dtype=np.float32),
                    qid=int(e["qid"]),
                )
            )
        return out

    def _save_questions(self):
        raw = []
        for q in self.questions:
            raw.append(
                dict(
                    msg_id=q.msg_id,
                    channel_id=q.channel_id,
                    author_id=q.author_id,
                    text=q.text,
                    ts=q.ts,
                    vec=q.vec.tolist(),
                )
            )
        json.dump(raw, open(Q_DB, "w", encoding="utf-8"), indent=2)

    def _save_answers(self):
        raw = []
        for a in self.answers:
            raw.append(
                dict(
                    msg_id=a.msg_id,
                    channel_id=a.channel_id,
                    author_id=a.author_id,
                    text=a.text,
                    ts=a.ts,
                    vec=a.vec.tolist(),
                    qid=a.qid,
                )
            )
        json.dump(raw, open(A_DB, "w", encoding="utf-8"), indent=2)

    def _load_all(self):
        self.questions = self._load_questions()
        self.answers = self._load_answers()

        self.q_by_id = {q.msg_id: q for q in self.questions}

        self.answers_by_qid.clear()
        for a in self.answers:
            self.answers_by_qid.setdefault(a.qid, []).append(a)

        self.recent_q_by_channel.clear()
        for q in self.questions:
            self.recent_q_by_channel.setdefault(q.channel_id, []).append(q.msg_id)

        # rebuild ANN index from disk
        self.q_index = SRPIndex(dim=self.dim, n_planes=self.q_index.n_planes, n_bands=self.q_index.n_bands)
        for q in self.questions:
            self.q_index.add(int(q.msg_id), q.vec)

    # ---------- question API ----------
    def add_question(self, msg_id: int, channel_id: int, author_id: int, text: str, ts: Optional[float] = None):
        ts = ts if ts is not None else time.time()
        v = embed(text)

        q = QItem(msg_id=msg_id, channel_id=channel_id, author_id=author_id, text=text, ts=ts, vec=v)
        self.questions.append(q)
        self.q_by_id[msg_id] = q
        self.recent_q_by_channel.setdefault(channel_id, []).append(msg_id)

        self.q_index.add(int(msg_id), v)
        self._save_questions()

    def search_questions(self, text: str, top_k: int = 5, min_sim: float = 0.75) -> List[Tuple[QItem, float]]:
        if len(self.questions) == 0:
            return []

        qv = embed(text)

        # 1) ANN candidate retrieval
        cand_ids = self.q_index.candidates(qv)

        # Safety fallback: for tiny datasets or if no bucket hit, scan all
        if not cand_ids or len(self.questions) <= 200:
            cand_ids = list(self.q_by_id.keys())

        # 2) Exact re-ranking by cosine
        scored: List[Tuple[QItem, float]] = []
        for mid in cand_ids:
            qi = self.q_by_id.get(int(mid))
            if qi is None:
                continue
            sim = cosine_sim(qv, qi.vec)
            if sim >= min_sim:
                scored.append((qi, float(sim)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ---------- answer API ----------
    def _time_decay(self, dt_seconds: float, tau: float = 300.0) -> float:
        return float(np.exp(-dt_seconds / tau))

    def _link_answer(self, answer_vec: np.ndarray, channel_id: int, ts: float) -> Optional[int]:
        # Search among the most recent questions in this channel
        candidates = self.recent_q_by_channel.get(channel_id, [])[-30:]  # last 30 questions

        best_qid = None
        best_score = -1e9

        for qid in reversed(candidates):
            q = self.q_by_id.get(int(qid))
            if q is None:
                continue

            sim = cosine_sim(answer_vec, q.vec)
            dt = abs(ts - q.ts)

            # Hybrid score: similarity + recency (explainable + robust)
            score = 0.7 * sim + 0.3 * self._time_decay(dt)

            if score > best_score:
                best_score = score
                best_qid = int(qid)

        # Require a minimum combined score (prevents dumb links)
        if best_qid is not None and best_score >= 0.55:
            return best_qid
        return None

    def add_answer(
        self,
        msg_id: int,
        channel_id: int,
        author_id: int,
        text: str,
        reply_to_msg_id: Optional[int] = None,
        ts: Optional[float] = None,
    ) -> Optional[int]:
        ts = ts if ts is not None else time.time()
        v = embed(text)

        # If user replied directly to a known question, link exactly.
        if reply_to_msg_id is not None and int(reply_to_msg_id) in self.q_by_id:
            qid = int(reply_to_msg_id)
        else:
            # Otherwise guess-link using similarity+time among recent questions in channel.
            qid = self._link_answer(v, channel_id, ts)

        if qid is None:
            return None

        a = AItem(msg_id=msg_id, channel_id=channel_id, author_id=author_id, text=text, ts=ts, vec=v, qid=qid)
        self.answers.append(a)
        self.answers_by_qid.setdefault(qid, []).append(a)
        self._save_answers()
        return qid

    def get_best_answer(self, q: QItem, max_len: int = 800) -> Optional[str]:
        answers = self.answers_by_qid.get(q.msg_id, [])
        if not answers:
            return None

        # choose answer most aligned to question in embedding space
        best = None
        best_sim = -1e9
        for a in answers:
            s = cosine_sim(q.vec, a.vec)
            if s > best_sim:
                best_sim = s
                best = a

        if best is None:
            return None

        txt = best.text.strip()
        if len(txt) > max_len:
            txt = txt[:max_len] + "…"
        return txt
