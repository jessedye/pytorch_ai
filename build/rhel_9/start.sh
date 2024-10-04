python3 -m venv gpt-env
source gpt-env/bin/activate

uvicorn ai.py:app --host 0.0.0.0 --port 8000
