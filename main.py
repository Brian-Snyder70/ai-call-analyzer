from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import sys
from pathlib import Path

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment.")

client = OpenAI(api_key=api_key)

OUTPUT_JSON = "analysis_report.json"
OUTPUT_TRANSCRIPT = "transcript_output.txt"


def read_transcript(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def transcribe_audio(file_path: str) -> str:
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file
        )

    text = getattr(transcription, "text", None)
    if not text:
        raise ValueError("Transcription response did not include text.")
    return text


def analyze_transcript(transcript: str) -> str:
    prompt = f"""
You are a contact center QA assistant.

Return ONLY valid JSON with this structure:

{{
  "summary": "Brief summary of the call",
  "customer_intent": "Primary customer goal",
  "sentiment": "Overall sentiment",
  "qa_observations": [
    "Observation 1",
    "Observation 2"
  ],
  "coaching_suggestions": [
    "Suggestion 1",
    "Suggestion 2"
  ],
  "retention_risk": "low | medium | high",
  "qa_score": 0,
  "call_disposition": "resolved | unresolved | follow_up_required",
  "escalation_flag": true
  "next_best_action": "Recommended next step for the agent or business"
}}

Scoring guidelines:
- qa_score: 0–100 based on professionalism, empathy, resolution
- escalation_flag: true if customer frustration or churn risk is present
- call_disposition:
  - resolved = issue handled
  - unresolved = not handled
  - follow_up_required = needs additional action

Transcript:
{transcript}
"""

    response = client.responses.create(
        model="gpt-5.4",
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "call_analysis",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "summary": {"type": "string"},
                        "customer_intent": {"type": "string"},
                        "sentiment": {"type": "string"},
                        "qa_observations": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "coaching_suggestions": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "retention_risk": {
                            "type": "string",
                            "enum": ["low", "medium", "high"]
                        },
                        "qa_score": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100
                        },
                        "call_disposition": {
                            "type": "string",
                            "enum": ["resolved", "unresolved", "follow_up_required"]
                        },
                        "escalation_flag": {
                            "type": "boolean"
                        }
                    },
                    "required": [
                        "summary",
                        "customer_intent",
                        "sentiment",
                        "qa_observations",
                        "coaching_suggestions",
                        "retention_risk",
                        "qa_score",
                        "call_disposition",
                        "escalation_flag"
                    ]
                },
                "strict": True
            }
        }
    )

    return response.output_text


def save_text(text: str, file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text)


def save_json(report_text: str, file_path: str) -> None:
    try:
        data = json.loads(report_text)
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(report_text)


def get_input_content(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()

    if ext in [".wav", ".mp3", ".m4a", ".webm"]:
        print(f"Transcribing audio file: {file_path}")
        transcript = transcribe_audio(file_path)
        save_text(transcript, OUTPUT_TRANSCRIPT)
        print(f"Transcript saved to {OUTPUT_TRANSCRIPT}")
        return transcript

    if ext == ".txt":
        print(f"Reading transcript file: {file_path}")
        return read_transcript(file_path)

    raise ValueError(f"Unsupported file type: {ext}")


def main() -> None:
    input_file = sys.argv[1] if len(sys.argv) > 1 else "sample_call.txt"

    transcript = get_input_content(input_file)
    print("Analyzing transcript...")

    report = analyze_transcript(transcript)
    save_json(report, OUTPUT_JSON)

    print(f"Analysis complete. JSON report saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()