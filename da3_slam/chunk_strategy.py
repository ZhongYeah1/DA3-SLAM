"""Chunk strategies for batching keyframes before DA3 inference."""

from dataclasses import dataclass


@dataclass
class ChunkConfig:
    """Configuration for how keyframes are batched into chunks."""
    chunk_size: int = 17       # total frames per chunk (including overlap)
    overlap: int = 1           # frames carried from previous chunk
    max_vram_gb: float = 24.0  # VRAM budget for memory guard

    @property
    def new_frames_per_chunk(self) -> int:
        return self.chunk_size - self.overlap


# Preset configurations for each strategy
STRATEGY_CONFIGS = {
    "baseline": ChunkConfig(chunk_size=17, overlap=1),
    "refine": ChunkConfig(chunk_size=17, overlap=1),
    "large-refine": ChunkConfig(chunk_size=80, overlap=15),
}


def estimate_vram_gb(num_frames: int, model_name: str = "da3-small") -> float:
    """Estimate VRAM needed for a DA3 forward pass.

    Based on empirical measurements: ~0.05GB per frame at 504px + base cost.
    """
    base_gb = {
        "da3-small": 4.0,
        "da3-base": 5.0,
        "da3-large": 8.0,
        "da3-giant": 12.0,
        "da3nested-giant-large": 16.0,
    }.get(model_name, 6.0)
    per_frame_gb = 0.05
    return base_gb + num_frames * per_frame_gb


def safe_chunk_size(requested: int, model_name: str, max_vram_gb: float) -> int:
    """Reduce chunk size if it would exceed VRAM budget."""
    while requested > 2 and estimate_vram_gb(requested, model_name) > max_vram_gb:
        requested = requested - 5
    return max(requested, 2)
