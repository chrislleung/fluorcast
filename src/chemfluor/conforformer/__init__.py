"""Local ConforFormer conformer-cache utilities.

This package intentionally starts with conformer generation and cache-key
building only. Encoder preprocessing, model loading, pooling, and FluorCast
feature integration are later implementation stages.
"""

from .config import ConformerGenerationConfig
from .conformers import generate_conformer_cache_record

__all__ = ["ConformerGenerationConfig", "generate_conformer_cache_record"]
