FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ASOFCAST_ARTIFACTS=/models \
    ASOFCAST_RUNTIME_DIR=/runtime \
    ASOFCAST_SERVING_BACKEND=onnx \
    PORT=8000

WORKDIR /app

RUN groupadd --system asofcast \
    && useradd --system --gid asofcast --home-dir /app --shell /usr/sbin/nologin asofcast

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --no-cache-dir 'torch==2.10.0' \
      --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir '.[optimize]'

USER asofcast
EXPOSE 8000

CMD ["sh", "-c", "python -m asofcast serve --artifacts \"${ASOFCAST_ARTIFACTS}\" --host 0.0.0.0 --port \"${PORT}\""]
