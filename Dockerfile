# syntax=docker/dockerfile:1.7
#
# ModuleVoiceAgent runtime: Asterisk chan_websocket media server with GigaAM STT, LLM, Piper and Silero TTS.
#   docker build -t voice-agent-runtime .
#
# CPU only (x86_64 and arm64). Models are downloaded into the /models volume on first start
# (or ahead of time: docker compose run --rm runtime download-models).

ARG PYTHON_IMAGE=python:3.12-slim-bookworm

FROM ${PYTHON_IMAGE} AS build
RUN python -m pip install --no-cache-dir uv==0.11.28
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_NO_CACHE=1
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
WORKDIR /src
COPY pyproject.toml ./
# Dependencies first, in their own layer: a code change rebuilds in seconds instead of reinstalling torch.
# CPU-only torch for the Silero voices (the default PyPI build for x86_64 bundles CUDA, ~2 GB more).
RUN uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
 && uv pip install -r pyproject.toml --extra speech
COPY README.md ./
COPY src ./src
RUN uv pip install --no-deps . \
 && python -c "import onnx_asr, onnxruntime, piper, torch, silero_stress, livekit.rtc, voice_agent_runtime; print('runtime', voice_agent_runtime.__version__)"

FROM ${PYTHON_IMAGE}
LABEL org.opencontainers.image.title="MikoPBX Voice Agent runtime" \
      org.opencontainers.image.description="AI voice secretary runtime for MikoPBX ModuleVoiceAgent"
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 ca-certificates tini \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --uid 10002 --home-dir /models --shell /usr/sbin/nologin agent \
 && mkdir -p /models /config /data \
 && chown agent:agent /models /data
COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VA_MODELS_DIR=/models \
    VA_TENANTS_FILE=/config/tenants.json \
    HF_HOME=/models/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TQDM_DISABLE=1 \
    OMP_NUM_THREADS=1
USER agent
VOLUME ["/models", "/data"]
EXPOSE 8443
HEALTHCHECK --interval=30s --timeout=10s --start-period=15m --retries=3 CMD ["voice-agent-runtime", "healthcheck"]
ENTRYPOINT ["tini", "--", "voice-agent-runtime"]
CMD ["serve"]
