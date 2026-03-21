# AI Call Analyzer

FastAPI-based demo that transcribes or ingests call transcripts, sends them to OpenAI for QA scoring, and renders structured insights in a single-page UI.

## Features

- Upload audio (`.wav`, `.mp3`, `.m4a`, `.webm`) or paste/pull text transcripts
- Whisper (GPT-4o mini) transcription + structured JSON analysis via the Responses API
- Downloadable JSON report plus quick-glance metrics (intent, sentiment, QA score, disposition, etc.)
- Ready-to-deploy Render blueprint (`render.yaml`) and local CLI runner (`main.py`)

## Requirements

- Python 3.11.x
- OpenAI API key with access to `gpt-4o-mini-transcribe` and `gpt-5.4`

## Local Development

```bash
python -m venv .venv
.venv\Scripts\activate        # or source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
set OPENAI_API_KEY=sk-...      # use export on macOS/Linux
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Visit <http://localhost:8000> and use the UI. For CLI-only runs:

```bash
python main.py sample_call.txt   # or pass an audio file path
```

Outputs land in `analysis_report.json` (structured results) and `transcript_output.txt` when the input was audio.

## Deploying on Render

1. Push this repo to GitHub.
2. Either:
   - Use **New → Blueprint** in Render and point it at `render.yaml`, or
   - Create a Web Service manually, set the build command to `pip install -r requirements.txt`, and the start command to `uvicorn api:app --host 0.0.0.0 --port $PORT`.
3. Add `OPENAI_API_KEY` under the service’s **Environment** tab.
4. Trigger a deploy (or enable auto-deploy).

## Project Layout

```
├── api.py              # FastAPI HTTP interface + static UI serving
├── main.py             # CLI helpers: transcription + analysis pipeline
├── static/             # index.html + Styles.css for the front-end
├── sample_call.txt     # Starter transcript for demos/tests
├── render.yaml         # Render blueprint for painless deploys
├── requirements.txt    # Runtime dependencies
├── pyproject.toml      # Project metadata (Python 3.11)
└── README.md
```

## Environment Variables

| Name | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Required. Used by both CLI + FastAPI routes to call OpenAI APIs. |

## Next Ideas

- Persist past analyses for download/history
- Add auth or simple rate limiting before sharing publicly
- Swap in smaller/cheaper models for low-stakes demo environments

Feel free to extend and keep shipping! 🤠
