from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import tempfile
import os
import json

from main import transcribe_audio, analyze_transcript

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# OpenAI's transcription endpoint rejects audio files larger than 25 MB.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024

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
        # The OpenAI client is synchronous; run it in a worker thread so a
        # slow request doesn't block every other request on the server.
        report = json.loads(await run_in_threadpool(analyze_transcript, transcript))
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

    too_large = HTTPException(
        status_code=413,
        detail=f"Audio file is too large. Maximum size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
    )

    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        raise too_large

    temp_path = None

    try:
        # Copy the upload to disk in chunks instead of loading it all into
        # memory, and stop as soon as it goes over the limit.
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_path = temp_file.name
            written = 0
            while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise too_large
                temp_file.write(chunk)

        if written == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        transcript = await run_in_threadpool(transcribe_audio, temp_path)
        report = json.loads(await run_in_threadpool(analyze_transcript, transcript))

        return {
            "filename": file.filename,
            "transcript": transcript,
            "analysis": report
        }

    except HTTPException:
        raise
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Model returned invalid JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)