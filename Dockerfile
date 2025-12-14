# VoxCPM TTS Docker Image
# 支持 VoxCPM 1.5 (44.1kHz) 和 VoxCPM 0.5B (16kHz)
FROM nvcr.io/nvidia/cuda:12.1.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 默认显存优化参数
ENV VOXCPM_GPU_MEMORY_UTILIZATION=0.4
ENV VOXCPM_MAX_NUM_SEQS=128
ENV VOXCPM_MAX_MODEL_LEN=2048

# 安装基础依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git ca-certificates ninja-build \
    && rm -rf /var/lib/apt/lists/*

# 安装 Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -afy

ENV PATH=/opt/conda/bin:$PATH

# 接受 Conda TOS 并创建环境
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -n voxcpm python=3.11 -y

# 安装 PyTorch 2.4.0
RUN /opt/conda/envs/voxcpm/bin/pip install --no-cache-dir \
    torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
RUN /opt/conda/envs/voxcpm/bin/pip install --no-cache-dir \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    fastapi uvicorn pydantic soundfile einops transformers aiohttp packaging

# 安装 flash-attn
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
ENV MAX_JOBS=4
RUN /opt/conda/envs/voxcpm/bin/pip install --no-cache-dir --no-build-isolation \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    flash-attn

# 复制代码
COPY nanovllm_voxcpm /app/nanovllm_voxcpm
COPY api_server /app/api_server
COPY pyproject.toml /app/

# 复制两个版本的模型（构建时可选）
# COPY VoxCPM1.5 /app/VoxCPM1.5
# COPY VoxCPM-0.5B /app/VoxCPM-0.5B

# 创建模型目录（运行时挂载）
RUN mkdir -p /app/VoxCPM1.5 /app/VoxCPM-0.5B /app/voices /app/frontend

# 安装项目
WORKDIR /app
RUN /opt/conda/envs/voxcpm/bin/pip install -e .

# 默认使用 VoxCPM 1.5，可通过环境变量切换
# VOXCPM_MODEL_PATH=/app/VoxCPM-0.5B 使用 0.5B 版本
ENV VOXCPM_MODEL_PATH=/app/VoxCPM1.5

EXPOSE 8081

CMD ["/opt/conda/envs/voxcpm/bin/uvicorn", "api_server.app:app", "--host", "0.0.0.0", "--port", "8081"]
