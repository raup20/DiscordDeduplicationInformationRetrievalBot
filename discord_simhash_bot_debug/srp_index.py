import numpy as np
from typing import Dict, List, Iterable, Tuple

class SRPIndex:
    """
    Signed Random Projection (SRP) index for cosine similarity.
    - Builds L random hyperplanes in R^d
    - Signature is L-bit sign(dot(v, r_i))
    - Uses banding to create buckets -> candidate retrieval
    """
    def __init__(self, dim: int, n_planes: int = 64, n_bands: int = 8, seed: int = 42):
        assert n_planes % n_bands == 0
        self.dim = dim
        self.n_planes = n_planes
        self.n_bands = n_bands
        self.band_size = n_planes // n_bands

        rng = np.random.default_rng(seed)
        self.planes = rng.normal(size=(n_planes, dim)).astype(np.float32)

        # buckets[band_id][band_key] -> list of item_ids
        self.buckets: List[Dict[int, List[int]]] = [dict() for _ in range(n_bands)]

    def _signature_bits(self, v: np.ndarray) -> np.ndarray:
        # v assumed normalized; sign(dot) gives bits
        proj = self.planes @ v  # (n_planes,)
        return (proj >= 0).astype(np.uint8)

    def _band_key(self, bits: np.ndarray, band_id: int) -> int:
        start = band_id * self.band_size
        end = start + self.band_size
        b = bits[start:end]
        key = 0
        for i, bit in enumerate(b):
            key |= (int(bit) << i)
        return key

    def add(self, item_id: int, v: np.ndarray):
        bits = self._signature_bits(v)
        for band in range(self.n_bands):
            key = self._band_key(bits, band)
            self.buckets[band].setdefault(key, []).append(item_id)

    def candidates(self, v: np.ndarray) -> List[int]:
        bits = self._signature_bits(v)
        cand = set()
        for band in range(self.n_bands):
            key = self._band_key(bits, band)
            for item_id in self.buckets[band].get(key, []):
                cand.add(item_id)
        return list(cand)
