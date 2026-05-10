# NyayaIQ - Indian Legal Intelligence

A prototype Indian legal research engine with:
- FastAPI backend for search and question answering
- Streamlit frontend for demo UI
- Offline fallback dataset for legal judgments
- PDF ingestion, retrieval, grounding, and conflict detection

## Setup

1. Create and activate a Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run the API backend

Start the FastAPI server on port `8002`:

```powershell
cd "s:\Company Assignments\Legal\NyayaIQ-Indian-Legal-Intelligence"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8002
```

The API will be available at `http://127.0.0.1:8002`.

## Run the Streamlit UI

Set the API base URL and start the UI on port `8502`:

```powershell
cd "s:\Company Assignments\Legal\NyayaIQ-Indian-Legal-Intelligence"
$env:API_BASE_URL = "http://127.0.0.1:8002"
python -m streamlit run app.py --server.headless true --server.port 8502
```

Open the UI in your browser at `http://127.0.0.1:8502`.

## Notes

- If Ollama or HF generation is unavailable, the system falls back to a template answer path.
- The app uses an offline curated judgment dataset when external APIs are not reachable.
- `app.py` respects `API_BASE_URL` from the environment for the backend endpoint.
