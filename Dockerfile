FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir poetry

COPY poetry.lock pyproject.toml /app/
COPY frontend /app/frontend
COPY backend /app/backend
COPY cosine_sim.pkl /app/cosine_sim.pkl
COPY movies.csv /app/movies.csv 

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --only main --no-root

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]