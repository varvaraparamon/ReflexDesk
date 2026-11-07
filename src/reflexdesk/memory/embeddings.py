"""Эмбеддеры L3: dense (BGE-M3), sparse (BM25), cross-encoder реранкер; ленивая загрузка."""

from __future__ import annotations

import threading
from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import models

from reflexdesk.config import get_settings

CUSTOM_DENSE = {
    "BAAI/bge-m3": dict(
        pooling=PoolingType.CLS, normalization=True, sources=ModelSource(hf="BAAI/bge-m3"), dim=1024,
        model_file="onnx/model.onnx", additional_files=["onnx/model.onnx_data", "onnx/Constant_7_attr__value"],
        size_in_gb=2.27,
    ),
}
_register_lock = threading.Lock()


def _ensure_registered(model: str) -> None:
    supported = {m["model"] for m in TextEmbedding.list_supported_models()}
    if model in supported or model not in CUSTOM_DENSE:
        return
    with _register_lock:
        if model not in {m["model"] for m in TextEmbedding.list_supported_models()}:
            TextEmbedding.add_custom_model(model=model, **CUSTOM_DENSE[model])


class Embedder:
    """Dense + sparse эмбеддинги и реранк одной точкой доступа."""

    def __init__(self, dense_model: str, sparse_model: str, reranker_model: str, cache_dir: str) -> None:
        _ensure_registered(dense_model)
        self._dense = TextEmbedding(dense_model, cache_dir=cache_dir)
        self._sparse = SparseTextEmbedding(sparse_model, cache_dir=cache_dir)
        self._reranker_model = reranker_model
        self._cache_dir = cache_dir
        self._reranker: TextCrossEncoder | None = None
        self.dim = len(next(iter(self._dense.embed(["probe"]))))

    def dense(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._dense.embed(texts)]

    def sparse(self, texts: list[str]) -> list[models.SparseVector]:
        return [
            models.SparseVector(indices=v.indices.tolist(), values=v.values.tolist())
            for v in self._sparse.embed(texts)
        ]

    def sparse_query(self, text: str) -> models.SparseVector:
        vector = next(iter(self._sparse.query_embed(text)))
        return models.SparseVector(indices=vector.indices.tolist(), values=vector.values.tolist())

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        if self._reranker is None:
            self._reranker = TextCrossEncoder(self._reranker_model, cache_dir=self._cache_dir)
        return [float(s) for s in self._reranker.rerank(query, documents)]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Синглтон эмбеддера по настройкам моделей."""
    settings = get_settings()
    return Embedder(
        settings.embedding_model, settings.sparse_model, settings.reranker_model,
        cache_dir=str(settings.data_dir / "models"),
    )
