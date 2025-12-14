# VoxCPM TTS Docker Image
FROM nvcr.io/nvidia/cuda:12.1.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# Default memory optimization settings
ENV VOXCPM_GPU_MEMORY_UTILIZATION=0.4
ENV VOXCPM_MAX_NUM_SEQS=128
ENV VOXCPM_MAX_MODEL_LEN=2048

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git ca-certificates ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -afy

ENV PATH=/opt/conda/bin:$PATH

# Create conda environment
RUN conda create -n voxcpm python=3.11 -y

# Install PyTorch 2.4.0
RUN /opt/conda/envs/voxcpm/bin/pip install --no-cache-dir \
    torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
RUN /opt/conda/envs/voxcpm/bin/pip install --no-cache-dir \
    fastapi uvicorn pydantic soundfile einops transformers aiohttp packaging

# Install flash-attn
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
ENV MAX_JOBS=4
RUN /opt/conda/envs/voxcpm/bin/pip install --no-cache-dir --no-build-isolation flash-attn

# Copy project files
WORKDIR /app
COPY nanovllm_voxcpm /app/nanovllm_voxcpm
COPY api_server /app/api_server
COPY pyproject.toml /app/

# Install project
RUN /opt/conda/envs/voxcpm/bin/pip install -e .

# Create directories for models and voices
RUN mkdir -p /app/VoxCPM1.5 /app/VoxCPM-0.5B /app/voices

EXPOSE 8081

# Start API server
CMD ["/opt/conda/envs/voxcpm/bin/uvicorn", "api_server.app:app", "--host", "0.0.0.0", "--port", "8081"]
