FROM python:3.12-slim

WORKDIR /app

RUN pip install poetry && poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock ./
RUN poetry install --no-dev --no-interaction --no-ansi

COPY alembic.ini ./
COPY alembic/ ./alembic/
COPY src/ ./src/
COPY scripts/ ./scripts/

EXPOSE 8000

CMD ["uvicorn", "easyweaver.main:app", "--host", "0.0.0.0", "--port", "8000"]
