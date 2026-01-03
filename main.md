Steps to run in GitHub Codespaces
Open your repo in Codespaces, then in the terminal install system packages:
Bash

sudo apt-get update
sudo apt-get install -y tesseract-ocr ffmpeg
Create a venv (optional but recommended):
Bash

python -m venv .venv
source .venv/bin/activate
Install Python dependencies:
Bash

pip install -U pip
pip install -r requirements.txt
Choose your GGUF model (env vars). Defaults are already set in the script, but you can override:
Bash

export FIRE_GGUF_REPO="bartowski/Qwen2.5-0.5B-Instruct-GGUF"
export FIRE_GGUF_FILE="Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
If the model is gated/private, also set:

Bash

export HF_TOKEN="your_hf_token"
Run the demo:
Bash

python fire.py --demo
Ask a one-off question:
Bash

python fire.py --ask "List available dataframes."
Run interactive chat:
Bash

python fire.py --chat
Attach multimodal files (optional):
Bash

python fire.py --ask "Summarize key points and cite sources." \
  --attach pdf:docs/paper.pdf:paper1 \
  --attach image:assets/figure.png:fig1
Audio example:

Bash

python fire.py --ask "What did they decide in the meeting?" \
  --attach audio:data/meeting.mp3:meeting1
Logging (optional):
Bash

python fire.py --chat --log
# logs written to: fire_outputs/fire_log.jsonl