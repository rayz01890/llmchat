# LLM Chat (minimal)

Quick local LLM chat proxy + static frontend.

Setup

1. Create a virtualenv and activate it.
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Set `OPENAI_API_KEY` in your environment or copy `.env.example` to `.env` and load it.

Run

```bash
uvicorn app.main:app --reload --port 8000
```

Open http://127.0.0.1:8000/ in your browser.

Notes

- The backend proxies `/api/chat` to the OpenAI-compatible `/chat/completions` endpoint.
- To test without an API key, open the root page to verify static files are served. Sending messages requires `OPENAI_API_KEY`.

Streamlit UI

Run the FastAPI server, then in a separate terminal start the Streamlit UI:

```bash
streamlit run streamlit_app.py
```

By default Streamlit will POST to `http://127.0.0.1:8000/api/chat`. Change the backend URL in the sidebar if needed.
