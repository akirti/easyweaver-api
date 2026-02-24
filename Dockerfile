FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for ibm_db and other native packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry && poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-interaction --no-ansi --no-root

COPY alembic.ini ./
COPY alembic/ ./alembic/
COPY src/ ./src/
COPY scripts/ ./scripts/

RUN poetry install --only main --no-interaction --no-ansi

EXPOSE 8001

CMD ["uvicorn", "easyweaver.main:app", "--host", "0.0.0.0", "--port", "8001"]
