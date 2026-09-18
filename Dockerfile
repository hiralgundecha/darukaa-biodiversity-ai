FROM python:3.11-slim

WORKDIR /app
COPY backend/requirements.txt .
# only the core runtime; the optional extras are installed separately if wanted
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pydantic numpy

COPY backend ./backend
COPY frontend ./frontend

ENV PYTHONPATH=/app/backend
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "backend"]
