from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import tempfile
import os
import json

from main import transcribe_audio, analyze_transcript

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="AI Call Analyzer UI")

# Serve static files like CSS/JS
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze-text")
async def analyze_text(payload: dict):
    transcript = payload.get("transcript")

    if not transcript:
        raise HTTPException(status_code=400, detail="Missing transcript")

    try:
        report = json.loads(analyze_transcript(transcript))
        return {
            "transcript": transcript,
            "analysis": report
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Model returned invalid JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text analysis failed: {str(e)}")


@app.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    ext = Path(file.filename).suffix.lower()
    if ext not in [".wav", ".mp3", ".m4a", ".webm"]:
        raise HTTPException(status_code=400, detail="Unsupported audio file type")

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name

        transcript = transcribe_audio(temp_path)
        report = json.loads(analyze_transcript(transcript))

        return {
            "filename": file.filename,
            "transcript": transcript,
            "analysis": report
        }

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Model returned invalid JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)