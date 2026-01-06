# Steps to run in GitHub Codespaces
Open your repo in Codespaces, then in the terminal install system packages:
```Bash

sudo apt-get update
sudo apt-get install -y tesseract-ocr ffmpeg
```
Create a venv (optional but recommended):
```Bash

python -m venv .venv
source .venv/bin/activate
```
Install Python dependencies:
```Bash

pip install -U pip
pip install -r requirements.txt
```
Choose your GGUF model (env vars). Defaults are already set in the script, but you can override:
```Bash

export FIRE_GGUF_REPO="bartowski/Qwen2.5-0.5B-Instruct-GGUF"
export FIRE_GGUF_FILE="Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
```
If the model is gated/private, also set:

```Bash

export HF_TOKEN="your_hf_token"
```
Run the demo:
```Bash

python fire.py --demo
```
Ask a one-off question:
```Bash

python fire.py --ask "List available dataframes."
```
Run interactive chat:
```Bash

python fire.py --chat
```
Attach multimodal files (optional):
```Bash

python fire.py --ask "Summarize key points and cite sources." \
  --attach pdf:docs/paper.pdf:paper1 \
  --attach image:assets/figure.png:fig1
```
Audio example:

```bash

python fire.py --ask "What did they decide in the meeting?" \
  --attach audio:data/meeting.mp3:meeting1
```
Logging (optional):

```bash

python fire.py --chat --log
# logs written to: fire_outputs/fire_log.jsonl
```

---

## Summary of Key Optimizations in V2

| # | Optimization | Memory Impact | Performance Impact |
|---|--------------|---------------|-------------------|
| 1 | Thread limiting via env vars | -5-10% RAM | Slight speed reduction |
| 2 | Reduced defaults (context, top-k) | -10-15% RAM | Faster |
| 3 | Model manager with unloading | **-30-50% RAM** | Slight overhead |
| 4 | Smaller chunks (600 vs 900) | -5% RAM | More chunks |
| 5 | Batch embedding processing | -10-20% RAM peaks | Same |
| 6 | Numpy fallback for FAISS, lazy BM25 | -5-10% RAM | Slightly slower |
| 7 | Smaller GGUF (Q2_K), reduced context | **-40-50% LLM RAM** | Slightly worse quality |
| 8 | Tiny Whisper, unload after use | **-80% ASR RAM** | Lower accuracy |
| 9 | Image resizing, streaming PDF | -20-30% peak RAM | Same |
| 10 | DataFrame eviction, sampling | -20-50% data RAM | May lose precision |

---

## Estimated Memory After Optimizations

| Component | Optimized RAM |
|-----------|---------------|
| Qwen2.5-0.5B Q2_K | ~250MB |
| all-MiniLM-L6-v2 | ~90MB |
| Whisper tiny (unloaded) | 0MB (loaded: ~75MB) |
| BLIP (unloaded) | 0MB (loaded: ~450MB) |
| Python + libs | ~400MB |
| **Total active** | **~750MB - 1.5GB** |

**it's feasible for 4GB RAM** with these optimizations.

---

## Environment Variables for Fine-Tuning

```bash
# For very constrained systems
export FIRE_LOW_MEMORY=1
export FIRE_N_CTX=1024          # Minimum viable context
export FIRE_N_BATCH=64          # Smaller batches
export FIRE_WHISPER_MODEL=tiny  # Smallest ASR
export FIRE_GGUF_FILE=Qwen2.5-0.5B-Instruct-Q2_K.gguf  # Aggressive quantization

# To disable multimodal entirely (saves most RAM)
export FIRE_DISABLE_MULTIMODAL=1
