#!/usr/bin/env python3
"""
FIRE (Fast Integrated Research Environment)
Single-file CPU-optimized multimodal MoE + LightRAG + Data Scientist tools.

Works well in GitHub Codespaces (Linux) with:
- pip deps installed
- system deps: tesseract-ocr, ffmpeg (for audio decoding)

Model:
- llama.cpp GGUF via llama-cpp-python
- set env vars FIRE_GGUF_REPO / FIRE_GGUF_FILE to choose the GGUF

Usage:
  python fire.py --demo
  python fire.py --ask "Profile df weather and plot temp over date."
  python fire.py --chat
  python fire.py --ask "Summarize" --attach pdf:docs/paper.pdf:paper1 --attach image:img.png:fig1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Avoid noisy tokenizer parallelism warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ----------------------------
# Runtime knobs (CPU-friendly defaults)
# ----------------------------
np.random.seed(7)
FIRE_MAX_TOOL_STEPS = int(os.environ.get("FIRE_MAX_TOOL_STEPS", "6"))
FIRE_TOPK_RAG = int(os.environ.get("FIRE_TOPK_RAG", "6"))
FIRE_MAX_CONTEXT_CHARS = int(os.environ.get("FIRE_MAX_CONTEXT_CHARS", "12000"))


# ----------------------------
# Lazy globals (heavy stuff)
# ----------------------------
_llm = None
_emb = None
_rag = None
_captioner = None
_whisper_asr = None

FIRE_MODELS: Dict[str, Any] = {}


# ----------------------------
# Utilities
# ----------------------------
def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks: List[str] = []
    i = 0
    while i < len(text):
        j = min(len(text), i + chunk_size)
        chunks.append(text[i:j])
        if j == len(text):
            break
        i = max(0, j - overlap)
    return chunks


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Embeddings
# ----------------------------
def get_embedder():
    global _emb
    if _emb is None:
        from sentence_transformers import SentenceTransformer

        emb_model_name = os.environ.get(
            "FIRE_EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        _emb = SentenceTransformer(emb_model_name, device="cpu")
    return _emb


def embed_texts(texts: List[str]) -> np.ndarray:
    emb = get_embedder()
    vecs = emb.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)


# ----------------------------
# LightRAG (hybrid BM25 + FAISS) with citations
# ----------------------------
@dataclass
class RagChunk:
    chunk_id: str
    source_id: str
    modality: str  # "text" | "pdf" | "image" | "audio"
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)


class LightRAG:
    def __init__(self, dim: int):
        import faiss
        from rank_bm25 import BM25Okapi

        self.dim = dim
        self.chunks: List[RagChunk] = []
        self._faiss = faiss.IndexFlatIP(dim)
        self._bm25 = None
        self._BM25Okapi = BM25Okapi
        self._bm25_corpus_tokens: List[List[str]] = []

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

        vecs = embed_texts([c.text for c in new_chunks])
        self._faiss.add(vecs)

        for c in new_chunks:
            self._bm25_corpus_tokens.append(self._tokenize(c.text))
            self.chunks.append(c)

        self._bm25 = self._BM25Okapi(self._bm25_corpus_tokens)

    def search(self, query: str, top_k: int = 6) -> List[Tuple[RagChunk, float]]:
        if not self.chunks:
            return []

        qv = embed_texts([query])
        D, I = self._faiss.search(qv, min(top_k * 4, len(self.chunks)))
        vec_hits = [
            (int(idx), float(score))
            for idx, score in zip(I[0], D[0])
            if idx != -1
        ]

        if self._bm25:
            bm_scores = self._bm25.get_scores(self._tokenize(query))
        else:
            bm_scores = np.zeros(len(self.chunks), dtype=np.float32)

        bm_scores = np.asarray(bm_scores, dtype=np.float32)
        bm_scores = (bm_scores - bm_scores.min()) / (bm_scores.max() - bm_scores.min() + 1e-6)

        scored: List[Tuple[RagChunk, float]] = []
        vec_map = {i: s for i, s in vec_hits}
        candidates = set(vec_map.keys())
        if len(candidates) < top_k * 3:
            candidates |= set(np.argsort(-bm_scores)[: top_k * 3].tolist())

        for i in candidates:
            v = float(vec_map.get(i, 0.0))
            b = float(bm_scores[i])
            score = 0.65 * v + 0.35 * b
            scored.append((self.chunks[i], score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def get_rag() -> LightRAG:
    global _rag
    if _rag is None:
        dim = int(get_embedder().get_sentence_embedding_dimension())
        _rag = LightRAG(dim=dim)
    return _rag


def format_rag_context(hits: List[Tuple[RagChunk, float]]) -> str:
    lines = []
    for c, s in hits:
        lines.append(f"[{c.source_id}:{c.chunk_id} | {c.modality} | score={s:.3f}] {c.text}")
    return "\n".join(lines)


# ----------------------------
# LLM (llama.cpp GGUF)
# ----------------------------
def get_llm():
    global _llm
    if _llm is not None:
        return _llm

    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    MODEL_REPO = os.environ.get("FIRE_GGUF_REPO", "bartowski/Qwen2.5-0.5B-Instruct-GGUF")
    MODEL_FILE = os.environ.get("FIRE_GGUF_FILE", "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf")

    print(f"[FIRE] Downloading GGUF: {MODEL_REPO} / {MODEL_FILE}", file=sys.stderr)
    try:
        gguf_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    except Exception as e:
        raise RuntimeError(
            "Failed to download GGUF model.\n"
            "Set env vars FIRE_GGUF_REPO and FIRE_GGUF_FILE to a valid GGUF.\n"
            f"Current: {MODEL_REPO} / {MODEL_FILE}\n"
            f"Original error: {repr(e)}"
        ) from e

    n_threads = max(2, os.cpu_count() or 2)
    n_ctx = int(os.environ.get("FIRE_N_CTX", "4096"))
    n_batch = int(os.environ.get("FIRE_N_BATCH", "256"))
    chat_format = os.environ.get("FIRE_CHAT_FORMAT")  # optional

    print(f"[FIRE] Loading llama.cpp (threads={n_threads}, ctx={n_ctx})", file=sys.stderr)
    kwargs = dict(
        model_path=gguf_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_batch=n_batch,
        verbose=False,
    )
    if chat_format:
        kwargs["chat_format"] = chat_format

    _llm = Llama(**kwargs)
    return _llm


def llm_chat(messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 700) -> str:
    llm = get_llm()

    # Prefer native chat completion when available
    if hasattr(llm, "create_chat_completion"):
        out = llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (out["choices"][0]["message"]["content"] or "").strip()

    # Fallback: simple concatenation
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


# ----------------------------
# Multimodal experts (lazy)
# ----------------------------
def get_captioner():
    global _captioner
    if _captioner is None:
        from transformers import pipeline

        _captioner = pipeline(
            task="image-to-text",
            model=os.environ.get("FIRE_CAPTION_MODEL", "Salesforce/blip-image-captioning-base"),
            device=-1,
        )
    return _captioner


def get_whisper_asr():
    global _whisper_asr
    if _whisper_asr is None:
        from faster_whisper import WhisperModel

        model_name = os.environ.get("FIRE_WHISPER_MODEL", "small")
        _whisper_asr = WhisperModel(model_name, device="cpu", compute_type="int8")
    return _whisper_asr


def ingest_pdf(path: str, source_id: Optional[str] = None) -> Dict[str, Any]:
    rag = get_rag()
    source_id = source_id or os.path.basename(path)
    text_parts: List[str] = []

    # Try PyMuPDF first
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(path)
        for page in doc:
            t = page.get_text("text") or ""
            if t.strip():
                text_parts.append(t)
        doc.close()
    except Exception:
        pass

    # Fallback: pdfplumber
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
    if not text:
        text = f"(No extractable text found in {path}. If scanned, OCR is needed.)"

    rag.add_text(source_id=source_id, text=text, modality="pdf", meta={"path": path})
    return {"source_id": source_id, "chars": len(text)}


def ingest_image(path: str, source_id: Optional[str] = None) -> Dict[str, Any]:
    rag = get_rag()
    source_id = source_id or os.path.basename(path)

    from PIL import Image

    im = Image.open(path).convert("RGB")

    # OCR
    ocr_text = ""
    try:
        import pytesseract

        ocr_text = pytesseract.image_to_string(im) or ""
    except Exception:
        ocr_text = ""

    # Caption
    caption = ""
    try:
        cap = get_captioner()(im)
        if isinstance(cap, list) and cap and "generated_text" in cap[0]:
            caption = cap[0]["generated_text"]
    except Exception:
        caption = ""

    merged = f"IMAGE_CAPTION: {caption}\nIMAGE_OCR: {ocr_text}".strip()
    if not merged:
        merged = "(No caption/OCR extracted.)"

    rag.add_text(source_id=source_id, text=merged, modality="image", meta={"path": path})
    return {"source_id": source_id, "caption": caption, "ocr_chars": len(ocr_text)}


def ingest_audio(path: str, source_id: Optional[str] = None) -> Dict[str, Any]:
    rag = get_rag()
    source_id = source_id or os.path.basename(path)

    seg_text: List[str] = []
    try:
        whisper_asr = get_whisper_asr()
        segments, info = whisper_asr.transcribe(path, beam_size=1, vad_filter=True)
        for seg in segments:
            seg_text.append(seg.text)
    except Exception as e:
        seg_text = [f"(ASR failed: {repr(e)})"]

    text = " ".join(seg_text).strip()
    rag.add_text(source_id=source_id, text=text, modality="audio", meta={"path": path})
    return {"source_id": source_id, "chars": len(text)}


# ----------------------------
# Data scientist toolkit (tools)
# ----------------------------
@dataclass
class DFHandle:
    name: str
    kind: str  # "pandas" | "polars"
    df: Any
    meta: Dict[str, Any] = field(default_factory=dict)


class DataRegistry:
    def __init__(self):
        self.frames: Dict[str, DFHandle] = {}

    def put_pandas(self, name: str, df: Any, meta: Optional[Dict[str, Any]] = None) -> str:
        self.frames[name] = DFHandle(name=name, kind="pandas", df=df, meta=meta or {})
        return name

    def put_polars(self, name: str, df: Any, meta: Optional[Dict[str, Any]] = None) -> str:
        self.frames[name] = DFHandle(name=name, kind="polars", df=df, meta=meta or {})
        return name

    def get(self, name: str) -> DFHandle:
        if name not in self.frames:
            raise KeyError(f"Unknown dataframe '{name}'. Available: {list(self.frames.keys())}")
        return self.frames[name]

    def list(self) -> List[str]:
        return list(self.frames.keys())


ds = DataRegistry()


def tool_load_csv(name: str, path: str) -> Dict[str, Any]:
    import pandas as pd

    df = pd.read_csv(path)
    ds.put_pandas(name, df, meta={"path": path})
    return {"df": name, "rows": int(df.shape[0]), "cols": int(df.shape[1])}


def tool_load_parquet(name: str, path: str) -> Dict[str, Any]:
    import polars as pl

    df = pl.read_parquet(path)
    ds.put_polars(name, df, meta={"path": path})
    return {"df": name, "rows": int(df.height), "cols": int(df.width)}


def tool_preview(df: str, n: int = 10) -> Dict[str, Any]:
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
        desc = d.describe(include="all").transpose()
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
    """
    Basic cleaning:
    - normalize column names
    - parse datetimes for date/time-ish columns
    - impute missing: numeric median, categorical most_frequent
    """
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
            # If a column is very weird, don't crash cleaning.
            pass

    ds.put_pandas(df, d, meta=dict(h.meta, cleaned="basic"))
    return {"df": df, "status": "cleaned_basic", "cols": list(d.columns)}


def tool_sql(df: str, query: str) -> Dict[str, Any]:
    import duckdb

    h = ds.get(df)
    d = h.df if h.kind == "pandas" else h.df.to_pandas()

    con = duckdb.connect(database=":memory:")
    con.register("t", d)
    out = con.execute(query).fetchdf()
    con.close()

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
    """
    Codespaces-friendly plotting:
    - saves an HTML file under ./fire_outputs/plots/
    - returns the path (you can open it in the Codespaces editor / browser preview)
    """
    import pandas as pd
    import plotly.express as px

    h = ds.get(df)
    d = h.df if isinstance(h.df, pd.DataFrame) else h.df.to_pandas()

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
    return {"status": "saved", "path": str(out_path), "kind": kind, "x": x, "y": y, "color": color}


def tool_train(
    df: str,
    target: str,
    task: str = "auto",
    test_size: float = 0.2,
    model: str = "auto",
) -> Dict[str, Any]:
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    h = ds.get(df)
    d = h.df if isinstance(h.df, pd.DataFrame) else h.df.to_pandas()

    if target not in d.columns:
        raise KeyError(f"Target '{target}' not in columns: {list(d.columns)}")

    X = d.drop(columns=[target])
    y = d[target]

    if task == "auto":
        # heuristic
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
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ]
    )

    if model == "auto":
        est = LogisticRegression(max_iter=200) if task == "classification" else LinearRegression()
    else:
        if task == "classification" and model == "rf":
            est = RandomForestClassifier(n_estimators=200, random_state=7)
        elif task == "regression" and model == "rf":
            est = RandomForestRegressor(n_estimators=200, random_state=7)
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
        if hasattr(pipe, "predict_proba"):
            try:
                if getattr(y, "nunique", lambda: 0)() == 2:
                    proba = pipe.predict_proba(Xte)[:, 1]
                    metrics["roc_auc"] = float(roc_auc_score(yte, proba))
            except Exception:
                pass
    else:
        metrics["rmse"] = float(math.sqrt(mean_squared_error(yte, pred)))
        metrics["r2"] = float(r2_score(yte, pred))

    model_id = f"model_{df}_{target}_{str(uuid.uuid4())[:4]}"
    FIRE_MODELS[model_id] = pipe
    return {
        "model_id": model_id,
        "task": task,
        "metrics": metrics,
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
    }


def tool_list_dataframes() -> Dict[str, Any]:
    return {"dataframes": ds.list()}


ToolFn = Callable[..., Dict[str, Any]]
TOOLS: Dict[str, Dict[str, Any]] = {
    "load_csv": {"fn": tool_load_csv, "desc": "Load a CSV into the registry.", "args": {"name": "str", "path": "str"}},
    "load_parquet": {"fn": tool_load_parquet, "desc": "Load a Parquet into the registry.", "args": {"name": "str", "path": "str"}},
    "list_dataframes": {"fn": tool_list_dataframes, "desc": "List registered dataframes.", "args": {}},
    "preview": {"fn": tool_preview, "desc": "Preview first N rows.", "args": {"df": "str", "n": "int"}},
    "profile": {"fn": tool_profile, "desc": "Schema + missingness + describe.", "args": {"df": "str"}},
    "clean_basic": {"fn": tool_clean_basic, "desc": "Basic cleaning (impute, normalize columns).", "args": {"df": "str", "strategy": "str"}},
    "sql": {"fn": tool_sql, "desc": "Run DuckDB SQL against dataframe as table 't'.", "args": {"df": "str", "query": "str"}},
    "join": {"fn": tool_join, "desc": "Join two dataframes on keys.", "args": {"left": "str", "right": "str", "on": "list[str]", "how": "str"}},
    "plot": {"fn": tool_plot, "desc": "Save a plotly chart to HTML (Codespaces-friendly).", "args": {"df": "str", "kind": "str", "x": "str", "y": "str", "color": "str"}},
    "train": {"fn": tool_train, "desc": "Train baseline ML model.", "args": {"df": "str", "target": "str", "task": "str", "test_size": "float", "model": "str"}},
}


def tools_schema_text() -> str:
    lines = ["AVAILABLE_TOOLS (call by emitting JSON in a tool_call fenced block):"]
    for name, t in TOOLS.items():
        lines.append(f"- {name}: {t['desc']} args={t['args']}")
    return "\n".join(lines)


# ----------------------------
# Agent
# ----------------------------
SYSTEM_PROMPT = """You are FIRE, the Fast Integrated Research Environment assistant embedded in an IDE.

Rules:
- Be reproducible: when you need data operations or training, use tools.
- Never pretend you executed a tool. If you didn't call a tool, say it plainly.
- Keep answers practical: propose next steps and show exact tool calls when appropriate.
- If RAG context is provided, cite sources using [source_id:chunk_id].
- If the user asks for "reasoning", provide a brief plan + key checks. Do NOT invent numbers.

Tool calling format (STRICT):
Return a JSON object inside a fenced block like:

```tool_call
{"name":"profile","args":{"df":"mydf"}}
If no tool is needed, do not emit tool_call.
"""
def extract_tool_call(text: str) -> Optional[Dict[str, Any]]:
    # Parse fenced block: tool_call\n{...}\n
    m = re.search(r"tool_call\s*(\{.*?\})\s*", text, flags=re.DOTALL)
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
        return fn(**args)
    except TypeError as e:
        return {
            "error": f"Bad args for tool '{name}': {repr(e)}",
            "expected": TOOLS[name]["args"],
            "got": args,
        }
    except Exception as e:
        return {"error": f"Tool '{name}' failed: {repr(e)}"}

@dataclass
class Attachment:
    kind: str  # "pdf"|"image"|"audio"|"text"
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
            rag.add_text(
                source_id=sid,
                text=a.text,
                modality="text",
                meta={"note": "user_text_attachment"},
            )
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

def fire_agent(user_text: str, attachments: Optional[List["Attachment"]] = None) -> Dict[str, Any]:
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

# ----------------------------
# Logging (optional)
# ----------------------------
FIRE_LOG: List[Dict[str, Any]] = []


def fire_agent_logged(user_text: str, attachments: Optional[List[Attachment]] = None) -> Dict[str, Any]:
    out = fire_agent(user_text, attachments)
    FIRE_LOG.append(
        {
            "user": user_text,
            "attachments": [asdict(a) for a in (attachments or [])],
            "answer": out["answer"],
            "tool_trace": out["tool_trace"],
            "rag_used": out["rag_used"],
        }
    )
    return out


def export_logs(path: str = "fire_outputs/fire_log.jsonl") -> str:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in FIRE_LOG:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path

# ----------------------------
# CLI helpers
# ----------------------------
def parse_attach(spec: str) -> Attachment:
    """
    Format:
    kind:path[:source_id]
    Examples:
    pdf:docs/paper.pdf:paper1
    image:assets/fig.png:fig1
    audio:data/meeting.mp3:meeting1
    text:@notes.txt:labnote1 (text from file)
    text:"some inline note":note1
    """
    # Split only first ":" (kind)
    if ":" not in spec:
        raise ValueError(f"Bad --attach '{spec}'. Expected kind:path[:source_id].")
    kind, rest = spec.split(":", 1)
    kind = kind.strip().lower()
    source_id = None
    path_or_text = rest
    # Optional source_id at the end
    if rest.count(":") >= 1:
        # split last ":" as source_id
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
        # strip optional quotes
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

    resp = fire_agent("Profile the dataframe weather and clean missing values using tools, then plot temp_c over date.")
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
    print("FIRE chat. Type 'exit' to quit.\n", file=sys.stderr)
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
            print()
        if log and FIRE_LOG:
            out_path = export_logs()
            print(f"[log] wrote: {out_path}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="Run the built-in demo (no external files).")
    ap.add_argument("--ask", type=str, default=None, help="Ask a single question and exit.")
    ap.add_argument("--chat", action="store_true", help="Interactive chat REPL.")
    ap.add_argument(
        "--attach",
        action="append",
        default=[],
        help="Attach sources: kind:path[:source_id] (repeatable).",
    )
    ap.add_argument("--log", action="store_true", help="Log interactions to fire_outputs/fire_log.jsonl")
    args = ap.parse_args()
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