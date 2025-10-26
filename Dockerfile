FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# WORKDIR /app

COPY pyproject.toml /
COPY signaturize/ signaturize/

RUN uv sync

COPY app.py .env /

CMD ["uv", "run", "gradio", "app.py"]