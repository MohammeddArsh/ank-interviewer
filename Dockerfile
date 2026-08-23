FROM python:3.13-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


FROM python:3.13-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODE=openrouter \
    TEMP_DIR=/tmp

COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /app/logs \
    && chown -R app:app /app

COPY --chown=app:app app.py config.py requirements.txt ./
COPY --chown=app:app audio ./audio
COPY --chown=app:app brain ./brain
COPY --chown=app:app interview ./interview
COPY --chown=app:app static ./static

# Bake the default Moonshine ONNX weights into the image so containers
# never download models at boot. HF_HOME keeps the hub cache in one place.
ENV HF_HOME=/opt/models/hf
RUN python -c "from audio._vendor.moonshine_onnx import MoonshineOnnxModel; MoonshineOnnxModel(model_name='base')" \
    && chown -R app:app /opt/models

USER app

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
