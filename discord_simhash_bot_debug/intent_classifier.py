import math
import re
from dataclasses import dataclass
from typing import List, Literal, Tuple

import numpy as np
from embed_engine import embed, cosine_sim

Label = Literal["question", "answer", "other"]

@dataclass(frozen=True)
class IntentResult:
    label: Label
    confidence: float
    sims: Tuple[float, float, float]  # (score_q, score_a, score_o)

class IntentClassifier:
    """Embedding-space intent classification with multiple prototypes + light lexical features.

    Why this exists:
      - Pure '?'-logic is too naive.
      - A single prototype sentence is also too brittle.
      - This combines:
          (1) similarity to *multiple* prototype phrases per class (AADS-ish, explainable)
          (2) small lexical priors (question words, imperative verbs, etc.)

    Output:
      - label in {question, answer, other}
      - confidence (softmax probability)
      - raw class scores (q,a,o) for debugging/reporting
    """
    def __init__(self):
        # Prototype sets: keep them short + varied.
        q_phrases = [
            "How do I do something?",
            "What is the definition of something?",
            "Why does this happen?",
            "Can someone help me with this problem?",
            "Does anyone know how to fix this error?",
            "Is it possible to do X in Python?",
            "Where can I find documentation for this?",
            "I get an error, how do I solve it?",
        ]
        a_phrases = [
            "You can do it by following these steps.",
            "The solution is to install it and run this command.",
            "Try this fix: do X, then Y.",
            "Here is an explanation of what it means.",
            "To solve it, use this method.",
            "Use pip install ..., then restart.",
            "In short, the answer is ...",
        ]
        o_phrases = [
            "Nice!",
            "Thanks!",
            "Lol",
            "I agree.",
            "This is unrelated chatter.",
            "ok",
        ]

        self.p_questions = [embed(s) for s in q_phrases]
        self.p_answers = [embed(s) for s in a_phrases]
        self.p_other = [embed(s) for s in o_phrases]

        # Precompile regex for speed
        self._re_qword = re.compile(r"^(how|what|why|where|when|who|which|can|could|should|do|does|did|is|are|am|will|would)\b", re.I)

    @staticmethod
    def _softmax3(a: float, b: float, c: float) -> Tuple[float, float, float]:
        m = max(a, b, c)
        ea, eb, ec = math.exp(a - m), math.exp(b - m), math.exp(c - m)
        s = ea + eb + ec
        return (ea / s, eb / s, ec / s)

    @staticmethod
    def _topk_mean(values: List[float], k: int = 3) -> float:
        if not values:
            return -1e9
        values = sorted(values, reverse=True)
        k = min(k, len(values))
        return float(sum(values[:k]) / k)

    def _proto_score(self, v: np.ndarray, protos: List[np.ndarray]) -> float:
        sims = [cosine_sim(v, p) for p in protos]
        # Using mean of top-3 is smoother than max and reduces weird flips
        return self._topk_mean(sims, k=3)

    def _lexical_prior(self, text: str) -> Tuple[float, float, float]:
        """Small, explainable lexical nudges (NOT the main decision)."""
        t = text.strip().lower()

        q_prior = 0.0
        a_prior = 0.0
        o_prior = 0.0

        if t.endswith("?"):
            q_prior += 0.20
        if self._re_qword.search(t):
            q_prior += 0.15

        # Answer-ish: imperatives / instruction style
        if t.startswith(("use ", "try ", "run ", "install ", "just ", "you can ", "first ", "then ")):
            a_prior += 0.15
        if "http://" in t or "https://" in t:
            a_prior += 0.05  # many answers include links

        # Other-ish: very short reactions
        if len(t) <= 4:
            o_prior += 0.10

        return (q_prior, a_prior, o_prior)

    def classify(self, text: str) -> IntentResult:
        v = embed(text)

        # Prototype-based scores
        s_q = self._proto_score(v, self.p_questions)
        s_a = self._proto_score(v, self.p_answers)
        s_o = self._proto_score(v, self.p_other)

        # Lexical priors
        pq, pa, po = self._lexical_prior(text)

        # Combine: proto dominates; lexical only nudges
        score_q = s_q + pq
        score_a = s_a + pa
        score_o = s_o + po

        # Convert to probabilities for a "confidence" number
        p_q, p_a, p_o = self._softmax3(score_q, score_a, score_o)

        # Choose top label (by score), with margin gating
        triples = [("question", score_q, p_q), ("answer", score_a, p_a), ("other", score_o, p_o)]
        triples.sort(key=lambda x: x[1], reverse=True)
        top_label, top_score, top_p = triples[0]
        second_score = triples[1][1]
        margin = top_score - second_score

        # If it's too close, call it "other" (avoids spamming suggestions on answers)
        if margin < 0.08:
            return IntentResult("other", float(top_p), (float(score_q), float(score_a), float(score_o)))

        return IntentResult(top_label, float(top_p), (float(score_q), float(score_a), float(score_o)))
