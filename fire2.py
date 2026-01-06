#!/usr/bin/env python3
"""
FIRE (Fast Integrated Research Environment) - CPU/RAM Optimized Edition
Optimized for systems with 4GB RAM.

Key optimizations:
- Aggressive memory management with model unloading
- Smaller default models
- Reduced context/batch sizes
- Memory-efficient data structures
- Lazy imports and processing
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import time
import uuid
import weakref
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Iterator

import numpy as np

# ============================================================
# OPTIMIZATION 1: Environment defaults for low-memory systems
# ============================================================
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", str(min(4, os.cpu_count() or 2)))
os.environ.setdefault("MKL_NUM_THREADS", str(min(4, os.cpu_count() or 2)))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(min(4, os.cpu_count() or 2)))

# Limit numpy threads
try:
    import threadpoolctl
    threadpoolctl.threadpool_limits(limits=min(4, os.cpu_count() or 2))
except ImportError:
    pass

# ============================================================
# OPTIMIZATION 2: Reduced runtime defaults
# ============================================================
np.random.seed(7)
FIRE_MAX_TOOL_STEPS = int(os.environ.get("FIRE_MAX_TOOL_STEPS", "4"))  # Reduced from 6
FIRE_TOPK_RAG = int(os.environ.get("FIRE_TOPK_RAG", "4"))  # Reduced from 6
FIRE_MAX_CONTEXT_CHARS = int(os.environ.get("FIRE_MAX_CONTEXT_CHARS", "8000"))  # Reduced from 12000
FIRE_LOW_MEMORY = os.environ.get("FIRE_LOW_MEMORY", "1") == "1"  # Enable by default

# ============================================================
# OPTIMIZATION 3: Model manager with unloading capability
# ============================================================
class ModelManager:
    """Centralized model management with unloading support."""
    
    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._load_times: Dict[str, float] = {}
        
    def get(self, key: str) -> Optional[Any]:
        return self._models.get(key)
    
    def set(self, key: str, model: Any) -> None:
        self._models[key] = model
        self._load_times[key] = time.time()
        
    def unload(self, key: str) -> bool:
        """Unload a model and free memory."""
        if key in self._models:
            del self._models[key]
            if key in self._load_times:
                del self._load_times[key]
            gc.collect()
            return True
        return False
    
    def unload_all(self) -> None:
        """Unload all models."""
        self._models.clear()
        self._load_times.clear()
        gc.collect()
        
    def unload_except(self, keep: List[str]) -> None:
        """Unload all models except specified ones."""
        to_remove = [k for k in self._models if k not in keep]
        for k in to_remove:
            del self._models[k]
            self._load_times.pop(k, None)
        if to_remove:
            gc.collect()
            
    def memory_pressure_unload(self) -> None:
        """Unload least recently used models under memory pressure."""
        if FIRE_LOW_MEMORY and len(self._models) > 1:
            # Keep only the most essential model (LLM)
            self.unload_except(["llm"])

_mgr = ModelManager()
FIRE_MODELS: Dict[str, Any] = {}


# ============================================================
# OPTIMIZATION 4: Memory-efficient text chunking with generator
# ============================================================
def chunk_text_iter(text: str, chunk_size: int = 600, overlap: int = 80) -> Iterator[str]:
    """Generator-based chunking to avoid creating large lists."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return
    i = 0
    while i < len(text):
        j = min(len(text), i + chunk_size)
        yield text[i:j]
        if j == len(text):
            break
        i = max(0, j - overlap)

def chunk_text(text: str, chunk_size: int = 600, overlap: int = 80) -> List[str]:
    """Reduced chunk size (600 vs 900) for lower memory."""
    return list(chunk_text_iter(text, chunk_size, overlap))


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# OPTIMIZATION 5: Lighter embedding model + batch processing
# ============================================================
def get_embedder():
    cached = _mgr.get("emb")
    if cached is not None:
        return cached
    
    from sentence_transformers import SentenceTransformer
    
    # Use even smaller model for 4GB systems
    emb_model_name = os.environ.get(
        "FIRE_EMB_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"  # 22M params, ~90MB
        # Alternative for even lower memory: "sentence-transformers/paraphrase-MiniLM-L3-v2" (~60MB)
    )
    
    print(f"[FIRE] Loading embedder: {emb_model_name}", file=sys.stderr)
    emb = SentenceTransformer(emb_model_name, device="cpu")
    
    # OPTIMIZATION: Convert to half precision if possible (saves ~50% memory)
    if FIRE_LOW_MEMORY:
        try:
            emb.half()  # May not work on all CPU architectures
        except Exception:
            pass
            
    _mgr.set("emb", emb)
    return emb


def embed_texts(texts: List[str], batch_size: int = 16) -> np.ndarray:
    """Batch processing to control memory spikes."""
    if not texts:
        return np.array([], dtype=np.float32).reshape(0, 0)
        
    emb = get_embedder()
    
    # OPTIMIZATION: Process in smaller batches for memory control
    if len(texts) <= batch_size:
        vecs = emb.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)
    
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vecs = emb.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_vecs.append(np.asarray(vecs, dtype=np.float32))
        
    return np.vstack(all_vecs)


# ============================================================
# OPTIMIZATION 6: Memory-efficient RAG with optional FAISS
# ============================================================
@dataclass(slots=True)  # OPTIMIZATION: Use slots to reduce memory
class RagChunk:
    chunk_id: str
    source_id: str
    modality: str
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)


class LightRAG:
    """
    Memory-optimized RAG implementation.
    
    Optimizations:
    - Optional FAISS (falls back to numpy for small corpora)
    - Lazy BM25 initialization
    - Incremental indexing
    - Float16 vectors when possible
    """
    
    def __init__(self, dim: int, use_faiss: bool = True):
        self.dim = dim
        self.chunks: List[RagChunk] = []
        self._use_faiss = use_faiss
        self._faiss_index = None
        self._vectors: Optional[np.ndarray] = None
        self._bm25 = None
        self._bm25_corpus_tokens: List[List[str]] = []
        self._dirty_bm25 = False
        
    def _get_faiss_index(self):
        if self._faiss_index is None and self._use_faiss:
            try:
                import faiss
                # OPTIMIZATION: Use IVF index for large corpora (faster, slightly less accurate)
                if FIRE_LOW_MEMORY and len(self.chunks) > 1000:
                    # IVF index uses less memory for search
                    quantizer = faiss.IndexFlatIP(self.dim)
                    nlist = min(100, len(self.chunks) // 10)
                    self._faiss_index = faiss.IndexIVFFlat(quantizer, self.dim, max(1, nlist))
                else:
                    self._faiss_index = faiss.IndexFlatIP(self.dim)
            except ImportError:
                self._use_faiss = False
        return self._faiss_index

    def _tokenize(self, s: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9_]+", s.lower())

    def add_text(
        self,
        source_id: str,
        text: str,
        modality: str = "text",
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        meta = meta or {}
        parts = chunk_text(text)
        if not parts:
            return

        new_chunks: List[RagChunk] = []
        for p in parts:
            c = RagChunk(
                chunk_id=str(uuid.uuid4())[:8],
                source_id=source_id,
                modality=modality,
                text=p,
                meta=dict(meta),
            )
            new_chunks.append(c)

        # OPTIMIZATION: Batch embed and add
        vecs = embed_texts([c.text for c in new_chunks])
        
        if self._use_faiss:
            idx = self._get_faiss_index()
            if idx is not None:
                idx.add(vecs)
        else:
            # Numpy fallback for systems without FAISS
            if self._vectors is None:
                self._vectors = vecs
            else:
                self._vectors = np.vstack([self._vectors, vecs])

        for c in new_chunks:
            self._bm25_corpus_tokens.append(self._tokenize(c.text))
            self.chunks.append(c)

        self._dirty_bm25 = True  # Lazy BM25 rebuild

    def _ensure_bm25(self) -> None:
        """Lazy BM25 initialization."""
        if self._dirty_bm25 and self._bm25_corpus_tokens:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(self._bm25_corpus_tokens)
            self._dirty_bm25 = False

    def search(self, query: str, top_k: int = 4) -> List[Tuple[RagChunk, float]]:
        if not self.chunks:
            return []

        qv = embed_texts([query])
        
        # Vector search
        if self._use_faiss and self._faiss_index is not None:
            D, I = self._faiss_index.search(qv, min(top_k * 3, len(self.chunks)))
            vec_hits = [
                (int(idx), float(score))
                for idx, score in zip(I[0], D[0])
                if idx != -1
            ]
        elif self._vectors is not None:
            # Numpy fallback
            scores = np.dot(self._vectors, qv.T).flatten()
            top_indices = np.argsort(-scores)[:top_k * 3]
            vec_hits = [(int(i), float(scores[i])) for i in top_indices]
        else:
            vec_hits = []

        # BM25 scores
        self._ensure_bm25()
        if self._bm25:
            bm_scores = self._bm25.get_scores(self._tokenize(query))
        else:
            bm_scores = np.zeros(len(self.chunks), dtype=np.float32)

        bm_scores = np.asarray(bm_scores, dtype=np.float32)
        bm_max = bm_scores.max()
        bm_min = bm_scores.min()
        if bm_max > bm_min:
            bm_scores = (bm_scores - bm_min) / (bm_max - bm_min + 1e-6)

        # Combine scores
        scored: List[Tuple[RagChunk, float]] = []
        vec_map = {i: s for i, s in vec_hits}
        candidates = set(vec_map.keys())
        if len(candidates) < top_k * 2:
            candidates |= set(np.argsort(-bm_scores)[: top_k * 2].tolist())

        for i in candidates:
            if i >= len(self.chunks):
                continue
            v = float(vec_map.get(i, 0.0))
            b = float(bm_scores[i])
            score = 0.65 * v + 0.35 * b
            scored.append((self.chunks[i], score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
    
    def clear(self) -> None:
        """Clear all data and free memory."""
        self.chunks.clear()
        self._bm25_corpus_tokens.clear()
        self._bm25 = None
        self._faiss_index = None
        self._vectors = None
        gc.collect()


def get_rag() -> LightRAG:
    cached = _mgr.get("rag")
    if cached is not None:
        return cached
    
    dim = int(get_embedder().get_sentence_embedding_dimension())
    rag = LightRAG(dim=dim, use_faiss=True)
    _mgr.set("rag", rag)
    return rag


def format_rag_context(hits: List[Tuple[RagChunk, float]]) -> str:
    # OPTIMIZATION: Use list comprehension + join instead of repeated concatenation
    return "\n".join(
        f"[{c.source_id}:{c.chunk_id} | {c.modality} | score={s:.3f}] {c.text}"
        for c, s in hits
    )


# ============================================================
# OPTIMIZATION 7: LLM with aggressive memory settings
# ============================================================
def get_llm():
    cached = _mgr.get("llm")
    if cached is not None:
        return cached
    
    # OPTIMIZATION: Unload other heavy models before loading LLM
    if FIRE_LOW_MEMORY:
        _mgr.unload("captioner")
        _mgr.unload("whisper")
        gc.collect()

    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    # OPTIMIZATION: Use smaller quantization for 4GB systems
    MODEL_REPO = os.environ.get("FIRE_GGUF_REPO", "bartowski/Qwen2.5-0.5B-Instruct-GGUF")
    
    # Q2_K uses ~40% less memory than Q4_K_M
    MODEL_FILE = os.environ.get(
        "FIRE_GGUF_FILE", 
        "Qwen2.5-0.5B-Instruct-Q2_K.gguf" if FIRE_LOW_MEMORY else "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
    )

    print(f"[FIRE] Downloading GGUF: {MODEL_REPO} / {MODEL_FILE}", file=sys.stderr)
    try:
        gguf_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    except Exception as e:
        raise RuntimeError(
            "Failed to download GGUF model.\n"
            f"Set env vars FIRE_GGUF_REPO and FIRE_GGUF_FILE to a valid GGUF.\n"
            f"Current: {MODEL_REPO} / {MODEL_FILE}\n"
            f"Original error: {repr(e)}"
        ) from e

    # OPTIMIZATION: Limit threads to prevent memory spikes
    n_threads = min(4, max(2, os.cpu_count() or 2))
    
    # OPTIMIZATION: Reduced context for memory savings
    # 2048 context uses ~50% less KV cache memory than 4096
    n_ctx = int(os.environ.get("FIRE_N_CTX", "2048" if FIRE_LOW_MEMORY else "4096"))
    
    # OPTIMIZATION: Smaller batch size
    n_batch = int(os.environ.get("FIRE_N_BATCH", "128" if FIRE_LOW_MEMORY else "256"))
    
    chat_format = os.environ.get("FIRE_CHAT_FORMAT")

    print(f"[FIRE] Loading llama.cpp (threads={n_threads}, ctx={n_ctx}, batch={n_batch})", file=sys.stderr)
    
    kwargs = dict(
        model_path=gguf_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_batch=n_batch,
        verbose=False,
        # OPTIMIZATION: Disable memory mapping on low-memory systems for predictable RAM usage
        use_mmap=not FIRE_LOW_MEMORY,
        # OPTIMIZATION: Use memory locking only if we have enough RAM
        use_mlock=False,
    )
    if chat_format:
        kwargs["chat_format"] = chat_format

    llm = Llama(**kwargs)
    _mgr.set("llm", llm)
    return llm


def llm_chat(messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 512) -> str:
    """
    OPTIMIZATION: Reduced default max_tokens from 700 to 512
    """
    llm = get_llm()

    if hasattr(llm, "create_chat_completion"):
        out = llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (out["choices"][0]["message"]["content"] or "").strip()

    # Fallback
    prompt = ""
    for m in messages:
        role = m["role"].upper()
        prompt += f"{role}: {m['content']}\n"
    prompt += "ASSISTANT:"
    out = llm(
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["USER:", "SYSTEM:", "TOOL:"],
    )
    return out["choices"][0]["text"].strip()


# ============================================================
# OPTIMIZATION 8: Lighter multimodal models with unloading
# ============================================================
def get_captioner():
    cached = _mgr.get("captioner")
    if cached is not None:
        return cached
    
    # Unload LLM if under memory pressure
    if FIRE_LOW_MEMORY:
        _mgr.memory_pressure_unload()
    
    from transformers import pipeline

    # OPTIMIZATION: Use smaller model variant
    model_name = os.environ.get(
        "FIRE_CAPTION_MODEL",
        "Salesforce/blip-image-captioning-base"  # ~450MB
        # For even lower memory: "nlpconnect/vit-gpt2-image-captioning" (~500MB but faster)
    )
    
    print(f"[FIRE] Loading captioner: {model_name}", file=sys.stderr)
    captioner = pipeline(
        task="image-to-text",
        model=model_name,
        device=-1,  # CPU
    )
    _mgr.set("captioner", captioner)
    return captioner


def unload_captioner():
    """Explicitly unload captioner to free memory."""
    _mgr.unload("captioner")


def get_whisper_asr():
    cached = _mgr.get("whisper")
    if cached is not None:
        return cached
    
    if FIRE_LOW_MEMORY:
        _mgr.memory_pressure_unload()
    
    from faster_whisper import WhisperModel

    # OPTIMIZATION: Use tiny/base for low memory (small=~500MB, tiny=~75MB, base=~150MB)
    model_name = os.environ.get(
        "FIRE_WHISPER_MODEL",
        "tiny" if FIRE_LOW_MEMORY else "small"
    )
    
    print(f"[FIRE] Loading Whisper: {model_name}", file=sys.stderr)
    asr = WhisperModel(
        model_name, 
        device="cpu", 
        compute_type="int8",  # Quantized for CPU
        cpu_threads=min(4, os.cpu_count() or 2),
    )
    _mgr.set("whisper", asr)
    return asr


def unload_whisper():
    """Explicitly unload Whisper to free memory."""
    _mgr.unload("whisper")


# ============================================================
# OPTIMIZATION 9: Streaming PDF/file processing
# ============================================================
def ingest_pdf(path: str, source_id: Optional[str] = None) -> Dict[str, Any]:
    rag = get_rag()
    source_id = source_id or os.path.basename(path)
    text_parts: List[str] = []

    try:
        import fitz

        doc = fitz.open(path)
        for page in doc:
            t = page.get_text("text") or ""
            if t.strip():
                text_parts.append(t)
                # OPTIMIZATION: Process in batches to control memory
                if len(text_parts) >= 10:
                    text = "\n".join(text_parts)
                    rag.add_text(source_id=source_id, text=text, modality="pdf", meta={"path": path})
                    text_parts.clear()
        doc.close()
    except Exception:
        pass

    if not text_parts:
        try:
            import pdfplumber

            with pdfplumber.open(path) as pdf:
                for p in pdf.pages:
                    t = p.extract_text() or ""
                    if t.strip():
                        text_parts.append(t)
        except Exception:
            pass

    text = "\n".join(text_parts).strip()
    if text:
        rag.add_text(source_id=source_id, text=text, modality="pdf", meta={"path": path})
    elif not rag.chunks:  # No text extracted at all
        text = f"(No extractable text found in {path}.)"
        rag.add_text(source_id=source_id, text=text, modality="pdf", meta={"path": path})

    # Force garbage collection after PDF processing
    gc.collect()
    
    return {"source_id": source_id, "chars": len(text)}


def ingest_image(path: str, source_id: Optional[str] = None) -> Dict[str, Any]:
    rag = get_rag()
    source_id = source_id or os.path.basename(path)

    from PIL import Image

    # OPTIMIZATION: Resize large images to reduce memory
    im = Image.open(path).convert("RGB")
    max_dim = 1024
    if max(im.size) > max_dim:
        ratio = max_dim / max(im.size)
        new_size = (int(im.size[0] * ratio), int(im.size[1] * ratio))
        im = im.resize(new_size, Image.Resampling.LANCZOS)

    ocr_text = ""
    try:
        import pytesseract
        ocr_text = pytesseract.image_to_string(im) or ""
    except Exception:
        ocr_text = ""

    caption = ""
    try:
        cap = get_captioner()(im)
        if isinstance(cap, list) and cap and "generated_text" in cap[0]:
            caption = cap[0]["generated_text"]
        # OPTIMIZATION: Unload captioner after use
        if FIRE_LOW_MEMORY:
            unload_captioner()
    except Exception:
        caption = ""

    merged = f"IMAGE_CAPTION: {caption}\nIMAGE_OCR: {ocr_text}".strip()
    if not merged:
        merged = "(No caption/OCR extracted.)"

    rag.add_text(source_id=source_id, text=merged, modality="image", meta={"path": path})
    
    # Clean up
    del im
    gc.collect()
    
    return {"source_id": source_id, "caption": caption, "ocr_chars": len(ocr_text)}


def ingest_audio(path: str, source_id: Optional[str] = None) -> Dict[str, Any]:
    rag = get_rag()
    source_id = source_id or os.path.basename(path)

    seg_text: List[str] = []
    try:
        whisper_asr = get_whisper_asr()
        segments, info = whisper_asr.transcribe(
            path, 
            beam_size=1,  # Reduced for CPU
            vad_filter=True,
            condition_on_previous_text=False,  # Saves memory
        )
        for seg in segments:
            seg_text.append(seg.text)
        
        # OPTIMIZATION: Unload Whisper after use
        if FIRE_LOW_MEMORY:
            unload_whisper()
            
    except Exception as e:
        seg_text = [f"(ASR failed: {repr(e)})"]

    text = " ".join(seg_text).strip()
    rag.add_text(source_id=source_id, text=text, modality="audio", meta={"path": path})
    
    gc.collect()
    return {"source_id": source_id, "chars": len(text)}


# ============================================================
# OPTIMIZATION 10: Memory-efficient data toolkit
# ============================================================
@dataclass(slots=True)
class DFHandle:
    name: str
    kind: str
    df: Any
    meta: Dict[str, Any] = field(default_factory=dict)


class DataRegistry:
    """Memory-aware DataFrame registry."""
    
    def __init__(self, max_frames: int = 5):
        self.frames: Dict[str, DFHandle] = {}
        self.max_frames = max_frames
        self._access_order: List[str] = []

    def _update_access(self, name: str) -> None:
        if name in self._access_order:
            self._access_order.remove(name)
        self._access_order.append(name)
        
    def _evict_if_needed(self) -> None:
        """Evict oldest DataFrames if over limit."""
        while len(self.frames) >= self.max_frames and self._access_order:
            oldest = self._access_order.pop(0)
            if oldest in self.frames:
                del self.frames[oldest]
                gc.collect()

    def put_pandas(self, name: str, df: Any, meta: Optional[Dict[str, Any]] = None) -> str:
        self._evict_if_needed()
        self.frames[name] = DFHandle(name=name, kind="pandas", df=df, meta=meta or {})
        self._update_access(name)
        return name

    def put_polars(self, name: str, df: Any, meta: Optional[Dict[str, Any]] = None) -> str:
        self._evict_if_needed()
        self.frames[name] = DFHandle(name=name, kind="polars", df=df, meta=meta or {})
        self._update_access(name)
        return name

    def get(self, name: str) -> DFHandle:
        if name not in self.frames:
            raise KeyError(f"Unknown dataframe '{name}'. Available: {list(self.frames.keys())}")
        self._update_access(name)
        return self.frames[name]

    def list(self) -> List[str]:
        return list(self.frames.keys())
    
    def remove(self, name: str) -> bool:
        if name in self.frames:
            del self.frames[name]
            if name in self._access_order:
                self._access_order.remove(name)
            gc.collect()
            return True
        return False


# OPTIMIZATION: Limit number of DataFrames in memory
ds = DataRegistry(max_frames=5 if FIRE_LOW_MEMORY else 10)


def tool_load_csv(name: str, path: str, sample: Optional[int] = None) -> Dict[str, Any]:
    """
    OPTIMIZATION: Added optional sampling for large files.
    """
    import pandas as pd
    
    # OPTIMIZATION: Use chunked reading for large files
    file_size = Path(path).stat().st_size
    if file_size > 100_000_000 and sample is None:  # >100MB
        sample = 100000  # Auto-sample large files
        
    if sample:
        df = pd.read_csv(path, nrows=sample)
    else:
        df = pd.read_csv(path)
        
    ds.put_pandas(name, df, meta={"path": path, "sampled": sample})
    return {"df": name, "rows": int(df.shape[0]), "cols": int(df.shape[1]), "sampled": sample}


def tool_load_parquet(name: str, path: str, columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    OPTIMIZATION: Support column selection to reduce memory.
    """
    import polars as pl
    
    if columns:
        df = pl.read_parquet(path, columns=columns)
    else:
        df = pl.read_parquet(path)
        
    ds.put_polars(name, df, meta={"path": path})
    return {"df": name, "rows": int(df.height), "cols": int(df.width)}


def tool_preview(df: str, n: int = 5) -> Dict[str, Any]:
    """OPTIMIZATION: Reduced default preview from 10 to 5 rows."""
    h = ds.get(df)
    if h.kind == "pandas":
        return {"df": df, "preview": h.df.head(n).to_dict(orient="records")}
    return {"df": df, "preview": h.df.head(n).to_dicts()}


def tool_profile(df: str) -> Dict[str, Any]:
    import numpy as np
    import pandas as pd

    h = ds.get(df)
    if h.kind == "pandas":
        d = h.df
        missing = d.isna().sum().to_dict()
        dtypes = d.dtypes.astype(str).to_dict()
        # OPTIMIZATION: Limit describe to first 20 columns
        cols_to_describe = list(d.columns)[:20]
        desc = d[cols_to_describe].describe(include="all").transpose()
        desc = desc.replace({np.nan: None}).to_dict(orient="index")
        return {
            "df": df,
            "rows": int(d.shape[0]),
            "cols": int(d.shape[1]),
            "dtypes": dtypes,
            "missing": missing,
            "describe": desc,
        }

    d = h.df
    nulls = {c: int(d.get_column(c).null_count()) for c in d.columns}
    dtypes = {c: str(t) for c, t in zip(d.columns, d.dtypes)}
    return {"df": df, "rows": int(d.height), "cols": int(d.width), "dtypes": dtypes, "missing": nulls}


def tool_clean_basic(df: str, strategy: str = "auto") -> Dict[str, Any]:
    import pandas as pd

    h = ds.get(df)
    if h.kind != "pandas":
        d = h.df.to_pandas()
    else:
        d = h.df.copy()

    d.columns = [str(c).strip().replace(" ", "_") for c in d.columns]

    for c in d.columns:
        if re.search(r"(date|time)", str(c), re.IGNORECASE):
            try:
                d[c] = pd.to_datetime(d[c], errors="ignore")
            except Exception:
                pass

    for c in d.columns:
        try:
            if d[c].dtype.kind in "biufc":
                if d[c].isna().any():
                    d[c] = d[c].fillna(d[c].median(numeric_only=True))
            else:
                if d[c].isna().any():
                    mode = d[c].mode(dropna=True)
                    fill = mode.iloc[0] if len(mode) else ""
                    d[c] = d[c].fillna(fill)
        except Exception:
            pass

    ds.put_pandas(df, d, meta=dict(h.meta, cleaned="basic"))
    return {"df": df, "status": "cleaned_basic", "cols": list(d.columns)}


def tool_sql(df: str, query: str) -> Dict[str, Any]:
    import duckdb

    h = ds.get(df)
    d = h.df if h.kind == "pandas" else h.df.to_pandas()

    # OPTIMIZATION: Use temporary in-memory database
    out = duckdb.query(query.replace("t", f"d")).df()

    out_name = f"{df}_sql_{str(uuid.uuid4())[:4]}"
    ds.put_pandas(out_name, out, meta={"from": df, "query": query})
    return {"df": out_name, "rows": int(out.shape[0]), "cols": int(out.shape[1])}


def tool_join(left: str, right: str, on: List[str], how: str = "inner") -> Dict[str, Any]:
    import pandas as pd

    L = ds.get(left).df
    R = ds.get(right).df
    if not isinstance(L, pd.DataFrame):
        L = L.to_pandas()
    if not isinstance(R, pd.DataFrame):
        R = R.to_pandas()

    out = L.merge(R, on=on, how=how)
    out_name = f"join_{left}_{right}_{str(uuid.uuid4())[:4]}"
    ds.put_pandas(out_name, out, meta={"left": left, "right": right, "on": on, "how": how})
    return {"df": out_name, "rows": int(out.shape[0]), "cols": int(out.shape[1])}


def tool_plot(df: str, kind: str, x: str, y: Optional[str] = None, color: Optional[str] = None) -> Dict[str, Any]:
    import pandas as pd
    import plotly.express as px

    h = ds.get(df)
    d = h.df if isinstance(h.df, pd.DataFrame) else h.df.to_pandas()

    # OPTIMIZATION: Sample for large datasets to speed up plotting
    if len(d) > 10000:
        d = d.sample(n=10000, random_state=42)

    if kind == "line":
        fig = px.line(d, x=x, y=y, color=color)
    elif kind == "scatter":
        fig = px.scatter(d, x=x, y=y, color=color)
    elif kind == "bar":
        fig = px.bar(d, x=x, y=y, color=color)
    elif kind == "hist":
        fig = px.histogram(d, x=x, color=color)
    else:
        raise ValueError("kind must be one of: line, scatter, bar, hist")

    out_dir = Path("fire_outputs") / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{df}_{kind}_{str(uuid.uuid4())[:6]}.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    
    del fig
    gc.collect()
    
    return {"status": "saved", "path": str(out_path), "kind": kind, "x": x, "y": y, "color": color}


def tool_train(
    df: str,
    target: str,
    task: str = "auto",
    test_size: float = 0.2,
    model: str = "auto",
) -> Dict[str, Any]:
    """
    OPTIMIZATION: Use simpler models and smaller ensembles.
    """
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    h = ds.get(df)
    d = h.df if isinstance(h.df, pd.DataFrame) else h.df.to_pandas()

    if target not in d.columns:
        raise KeyError(f"Target '{target}' not in columns: {list(d.columns)}")

    # OPTIMIZATION: Sample large datasets for training
    if len(d) > 50000 and FIRE_LOW_MEMORY:
        d = d.sample(n=50000, random_state=7)

    X = d.drop(columns=[target])
    y = d[target]

    if task == "auto":
        try:
            task = "classification" if (y.dtype == object or y.nunique() <= 20) else "regression"
        except Exception:
            task = "regression"

    cat_cols = [c for c in X.columns if X[c].dtype == object]
    num_cols = [c for c in X.columns if X[c].dtype != object]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),  # Keep sparse
            ]), cat_cols),
        ],
        sparse_threshold=0.3,  # Use sparse matrices when beneficial
    )

    # OPTIMIZATION: Reduced estimators for RandomForest
    if model == "auto":
        est = LogisticRegression(max_iter=200) if task == "classification" else LinearRegression()
    else:
        n_estimators = 50 if FIRE_LOW_MEMORY else 200
        if task == "classification" and model == "rf":
            est = RandomForestClassifier(n_estimators=n_estimators, random_state=7, n_jobs=1)
        elif task == "regression" and model == "rf":
            est = RandomForestRegressor(n_estimators=n_estimators, random_state=7, n_jobs=1)
        else:
            raise ValueError("model must be 'auto' or 'rf'")

    pipe = Pipeline(steps=[("pre", pre), ("est", est)])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=float(test_size), random_state=7)
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)

    metrics: Dict[str, float] = {}
    if task == "classification":
        metrics["accuracy"] = float(accuracy_score(yte, pred))
        metrics["f1_macro"] = float(f1_score(yte, pred, average="macro"))
    else:
        metrics["rmse"] = float(math.sqrt(mean_squared_error(yte, pred)))
        metrics["r2"] = float(r2_score(yte, pred))

    model_id = f"model_{df}_{target}_{str(uuid.uuid4())[:4]}"
    FIRE_MODELS[model_id] = pipe
    
    # Clean up training data
    del Xtr, ytr
    gc.collect()
    
    return {
        "model_id": model_id,
        "task": task,
        "metrics": metrics,
        "n_train": int(len(X) - len(Xte)),
        "n_test": int(len(Xte)),
    }


def tool_list_dataframes() -> Dict[str, Any]:
    return {"dataframes": ds.list()}


def tool_free_memory() -> Dict[str, Any]:
    """New tool to manually free memory."""
    before = len(_mgr._models)
    _mgr.unload_except(["llm", "emb"])  # Keep essential models
    gc.collect()
    return {"status": "freed", "models_unloaded": before - len(_mgr._models)}


ToolFn = Callable[..., Dict[str, Any]]
TOOLS: Dict[str, Dict[str, Any]] = {
    "load_csv": {"fn": tool_load_csv, "desc": "Load a CSV into the registry.", "args": {"name": "str", "path": "str", "sample": "int (optional)"}},
    "load_parquet": {"fn": tool_load_parquet, "desc": "Load a Parquet into the registry.", "args": {"name": "str", "path": "str", "columns": "list[str] (optional)"}},
    "list_dataframes": {"fn": tool_list_dataframes, "desc": "List registered dataframes.", "args": {}},
    "preview": {"fn": tool_preview, "desc": "Preview first N rows.", "args": {"df": "str", "n": "int"}},
    "profile": {"fn": tool_profile, "desc": "Schema + missingness + describe.", "args": {"df": "str"}},
    "clean_basic": {"fn": tool_clean_basic, "desc": "Basic cleaning (impute, normalize columns).", "args": {"df": "str", "strategy": "str"}},
    "sql": {"fn": tool_sql, "desc": "Run DuckDB SQL against dataframe as table 'd'.", "args": {"df": "str", "query": "str"}},
    "join": {"fn": tool_join, "desc": "Join two dataframes on keys.", "args": {"left": "str", "right": "str", "on": "list[str]", "how": "str"}},
    "plot": {"fn": tool_plot, "desc": "Save a plotly chart to HTML.", "args": {"df": "str", "kind": "str", "x": "str", "y": "str", "color": "str"}},
    "train": {"fn": tool_train, "desc": "Train baseline ML model.", "args": {"df": "str", "target": "str", "task": "str", "test_size": "float", "model": "str"}},
    "free_memory": {"fn": tool_free_memory, "desc": "Free memory by unloading unused models.", "args": {}},
}


def tools_schema_text() -> str:
    lines = ["AVAILABLE_TOOLS (call by emitting JSON in a tool_call fenced block):"]
    for name, t in TOOLS.items():
        lines.append(f"- {name}: {t['desc']} args={t['args']}")
    return "\n".join(lines)


# ============================================================
# Agent (unchanged logic, but with memory management)
# ============================================================
SYSTEM_PROMPT = """You are FIRE, the Fast Integrated Research Environment assistant.

Rules:
- Be reproducible: use tools for data operations.
- Keep answers concise and practical.
- If RAG context is provided, cite sources using [source_id:chunk_id].
- Use free_memory tool if operations are slow.

Tool calling format:
```tool_call
{"name":"profile","args":{"df":"mydf"}}
"""

def extract_tool_call(text: str) -> Optional[Dict[str, Any]]:
  m = re.search(r"tool_call\s*({.?})\s", text, flags=re.DOTALL)
  if not m:
    return None
  raw = m.group(1).strip()
  try:
    obj = json.loads(raw)
    if isinstance(obj, dict) and "name" in obj and "args" in obj:
      return obj
  except Exception:
    return None
  return None

def run_tool(call: Dict[str, Any]) -> Dict[str, Any]:
  name = call.get("name")
  args = call.get("args", {}) or {}
  if name not in TOOLS:
    return {"error": f"Unknown tool '{name}'.", "available": list(TOOLS.keys())}
  fn = TOOLS[name]["fn"]
  try:
    result = fn(**args)
    gc.collect() # Clean up after each tool
    return result
  except TypeError as e:
    return {"error": f"Bad args for tool '{name}': {repr(e)}", "expected": TOOLS[name]["args"], "got": args}
  except Exception as e:
    return {"error": f"Tool '{name}' failed: {repr(e)}"}

@dataclass(slots=True)
class Attachment:
  kind: str
  path: Optional[str] = None
  text: Optional[str] = None
  source_id: Optional[str] = None
  
def moe_prepare_context(user_text: str, attachments: Optional[List[Attachment]]) -> Dict[str, Any]:
  rag = get_rag()
  ingest_reports: List[Dict[str, Any]] = []
  attachments = attachments or []
  for a in attachments:
    if a.kind == "text" and a.text:
        sid = a.source_id or f"text_{str(uuid.uuid4())[:6]}"
        rag.add_text(source_id=sid, text=a.text, modality="text", meta={"note": "user_text_attachment"})
        ingest_reports.append({"kind": "text", "source_id": sid, "chars": len(a.text)})
    elif a.kind == "pdf" and a.path:
        ingest_reports.append({"kind": "pdf", "report": ingest_pdf(a.path, a.source_id)})
    elif a.kind == "image" and a.path:
        ingest_reports.append({"kind": "image", "report": ingest_image(a.path, a.source_id)})
    elif a.kind == "audio" and a.path:
        ingest_reports.append({"kind": "audio", "report": ingest_audio(a.path, a.source_id)})
  hits = rag.search(user_text, top_k=FIRE_TOPK_RAG)
  ctx = format_rag_context(hits)
  if len(ctx) > FIRE_MAX_CONTEXT_CHARS:
      ctx = ctx[:FIRE_MAX_CONTEXT_CHARS] + "\n...(truncated)..."
  
  return {"ingested": ingest_reports, "rag_hits": hits, "rag_context": ctx}

def fire_agent(user_text: str, attachments: Optional[List[Attachment]] = None) -> Dict[str, Any]:
  prep = moe_prepare_context(user_text, attachments)
  tool_trace: List[Dict[str, Any]] = []
  messages: List[Dict[str, str]] = [
  {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + tools_schema_text()},
  ]
  if prep["rag_context"].strip():
    messages.append({"role": "system", "content": "RAG_CONTEXT:\n" + prep["rag_context"]})
  messages.append({"role": "user", "content": user_text})
  
  final_text = ""
  for _step in range(FIRE_MAX_TOOL_STEPS):
      assistant = llm_chat(messages)
      call = extract_tool_call(assistant)
  
      if not call:
          final_text = assistant
          break
  
      tool_out = run_tool(call)
      tool_trace.append({"call": call, "result": tool_out})
  
      messages.append({"role": "assistant", "content": assistant})
      messages.append({"role": "system", "content": "TOOL_RESULT:\n" + json.dumps(tool_out, ensure_ascii=False)})
  
      if "error" in tool_out:
          final_text = assistant + "\n\nTool error:\n" + json.dumps(tool_out, indent=2, ensure_ascii=False)
          break
  
  return {
      "answer": final_text.strip(),
      "tool_trace": tool_trace,
      "ingestion": prep["ingested"],
      "rag_used": [
          {"source_id": c.source_id, "chunk_id": c.chunk_id, "modality": c.modality, "score": float(s)}
          for c, s in prep["rag_hits"]
      ],
  }
#============================================================
#Logging
#============================================================
FIRE_LOG: List[Dict[str, Any]] = []

def fire_agent_logged(user_text: str, attachments: Optional[List[Attachment]] = None) -> Dict[str, Any]:
  out = fire_agent(user_text, attachments)
  FIRE_LOG.append({
  "user": user_text,
  "attachments": [asdict(a) for a in (attachments or [])],
  "answer": out["answer"],
  "tool_trace": out["tool_trace"],
  "rag_used": out["rag_used"],
  })
  return out

def export_logs(path: str = "fire_outputs/fire_log.jsonl") -> str:
  ensure_parent_dir(path)
  with open(path, "w", encoding="utf-8") as f:
    for row in FIRE_LOG:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")
  return path

#============================================================
#CLI
#============================================================
def parse_attach(spec: str) -> Attachment:
  if ":" not in spec:
    raise ValueError(f"Bad --attach '{spec}'. Expected kind:path[:source_id].")
  kind, rest = spec.split(":", 1)
  kind = kind.strip().lower()
  source_id = None
  path_or_text = rest
  if rest.count(":") >= 1:
      path_or_text, source_id = rest.rsplit(":", 1)
      path_or_text = path_or_text.strip()
      source_id = source_id.strip() or None
  else:
      path_or_text = path_or_text.strip()
  
  if kind == "text":
      if path_or_text.startswith("@"):
          p = path_or_text[1:]
          txt = Path(p).read_text(encoding="utf-8")
          return Attachment(kind="text", text=txt, source_id=source_id or os.path.basename(p))
      txt = path_or_text.strip()
      if (txt.startswith('"') and txt.endswith('"')) or (txt.startswith("'") and txt.endswith("'")):
          txt = txt[1:-1]
      return Attachment(kind="text", text=txt, source_id=source_id)
  
  if kind in ("pdf", "image", "audio"):
      p = str(Path(path_or_text).expanduser())
      return Attachment(kind=kind, path=p, source_id=source_id)
  
  raise ValueError(f"Unknown attachment kind '{kind}'. Use pdf|image|audio|text.")

def cmd_demo() -> None:
  import pandas as pd
  demo = pd.DataFrame({
  "date": pd.date_range("2024-01-01", periods=10, freq="D"),
  "temp_c": [24, 25, None, 26, 27, 30, 29, None, 28, 27],
  "city": ["A"] * 10,
  })
  ds.put_pandas("weather", demo, meta={"source": "synthetic"})
  resp = fire_agent("Profile the dataframe weather and clean missing values, then plot temp_c over date.")
  print(resp["answer"])
  print("\nTOOL TRACE:")
  print(json.dumps(resp["tool_trace"], indent=2, ensure_ascii=False))

def cmd_ask(question: str, attachments: List[Attachment], log: bool) -> None:
  out = fire_agent_logged(question, attachments) if log else fire_agent(question, attachments)
  print(out["answer"])
  if out["tool_trace"]:
    print("\nTOOL TRACE:")
    print(json.dumps(out["tool_trace"], indent=2, ensure_ascii=False))
  if log:
    out_path = export_logs()
    print(f"\n[log] wrote: {out_path}")

def cmd_chat(attachments: List[Attachment], log: bool) -> None:
  print("FIRE chat (optimized for 4GB). Type 'exit' to quit.\n", file=sys.stderr)
  while True:
    try:
      user_text = input("you> ").strip()
    except EOFError:
      break
    if not user_text:
      continue
    if user_text.lower() in ("exit", "quit"):
      break
    out = fire_agent_logged(user_text, attachments) if log else fire_agent(user_text, attachments)
    print("\n" + out["answer"] + "\n")
    if out["tool_trace"]:
      print("[tool_trace]")
      print(json.dumps(out["tool_trace"], indent=2, ensure_ascii=False))
    if log and FIRE_LOG:
      out_path = export_logs()
      print(f"[log] wrote: {out_path}", file=sys.stderr)

def main() -> None:
  ap = argparse.ArgumentParser(description="FIRE - CPU/RAM optimized for 4GB systems")
  ap.add_argument("--demo", action="store_true", help="Run built-in demo.")
  ap.add_argument("--ask", type=str, default=None, help="Ask a single question.")
  ap.add_argument("--chat", action="store_true", help="Interactive chat.")
  ap.add_argument("--attach", action="append", default=[], help="Attach sources: kind:path[:source_id]")
  ap.add_argument("--log", action="store_true", help="Log interactions.")
  ap.add_argument("--high-memory", action="store_true", help="Disable low-memory optimizations.")
  args = ap.parse_args()
  global FIRE_LOW_MEMORY
  if args.high_memory:
      FIRE_LOW_MEMORY = False
      os.environ["FIRE_LOW_MEMORY"] = "0"
  
  attachments: List[Attachment] = []
  for spec in args.attach:
      attachments.append(parse_attach(spec))
  
  if args.demo:
      cmd_demo()
      return
  if args.ask:
      cmd_ask(args.ask, attachments, log=args.log)
      return
  if args.chat:
      cmd_chat(attachments, log=args.log)
      return
  
  ap.print_help()

if __name__ == "__main__":
  main()
  
