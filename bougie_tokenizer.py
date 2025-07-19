import numpy as np
from typing import Tuple, Dict

class CandleTokenizer:
    """Tokenize normalized and quantized OHLCV candles."""

    def __init__(self, quant: float = 1e-4, norm_factor: float = 100.0):
        self.quant = quant
        self.norm_factor = norm_factor
        self.codebook: Dict[Tuple[float, ...], int] = {}
        self.inverse_codebook: Dict[int, Tuple[float, ...]] = {}

    def _normalize(self, candle: Tuple[float, float, float, float, float]) -> np.ndarray:
        return np.array(candle, dtype=np.float32) / self.norm_factor

    def _quantize(self, values: np.ndarray) -> Tuple[float, ...]:
        return tuple(np.round(values / self.quant) * self.quant)

    def encode(self, candle: Tuple[float, float, float, float, float]) -> int:
        """Return integer token for given candle."""
        key = self._quantize(self._normalize(candle))
        if key not in self.codebook:
            idx = len(self.codebook)
            self.codebook[key] = idx
            self.inverse_codebook[idx] = key
        return self.codebook[key]

    def decode(self, token: int) -> Tuple[float, float, float, float, float]:
        values = np.array(self.inverse_codebook[token]) * self.norm_factor
        return tuple(values.astype(float))
