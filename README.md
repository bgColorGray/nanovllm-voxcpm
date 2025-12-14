# Nano-vLLM-VoxCPM

An inference engine for VoxCPM based on Nano-vLLM.

## Features

- Faster than the pytorch implementation
- Support concurrent requests
- OpenAI compatible API server (see [api_server/](api_server/))
- Memory optimization with environment variables
- Performance benchmark tools (see [benchmark/](benchmark/))

## Installation

```bash
git clone https://github.com/bgColorGray/nanovllm-voxcpm.git
cd nanovllm-voxcpm
pip install -e .
```

## Quick Start

### 1. Docker Deployment (Recommended)

```bash
# Clone the repository
git clone https://github.com/bgColorGray/nanovllm-voxcpm.git
cd nanovllm-voxcpm

# Download VoxCPM model to ./VoxCPM1.5 or ./VoxCPM-0.5B

# Start with docker-compose
docker-compose up -d

# Or build and run manually
docker build -t voxcpm-tts .
docker run -d --gpus all \
  -p 8081:8081 \
  -v ./VoxCPM1.5:/app/VoxCPM1.5:ro \
  -e VOXCPM_GPU_MEMORY_UTILIZATION=0.4 \
  -e VOXCPM_MAX_NUM_SEQS=128 \
  voxcpm-tts
```

### 2. Basic Usage

See the [example.py](example.py) for a basic usage example.

### 3. OpenAI Compatible API Server

```bash
# Set environment variables (optional, use defaults if not set)
export VOXCPM_MODEL_PATH=/path/to/VoxCPM1.5
export VOXCPM_GPU_MEMORY_UTILIZATION=0.4
export VOXCPM_MAX_NUM_SEQS=128
export VOXCPM_MAX_MODEL_LEN=2048

# Start the server
cd api_server
uvicorn app:app --host 0.0.0.0 --port 8080
```

API endpoints:
- `POST /v1/audio/speech` - Generate speech (OpenAI compatible)
- `POST /v1/audio/speech/stream` - Streaming speech generation
- `POST /v1/voices` - Create voice clone
- `GET /health` - Health check

### 3. Multi-GPU Deployment

```bash
# Data parallel: 2 groups × 2 GPUs each
export CUDA_VISIBLE_DEVICES=0,1
export VOXCPM_DEVICES=0,1

# Tensor parallel: 1 group × 4 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VOXCPM_DEVICES=0,1,2,3
```

## Memory Optimization

### Recommended Configuration

```bash
VOXCPM_GPU_MEMORY_UTILIZATION=0.4  # GPU memory utilization (0.4-0.9)
VOXCPM_MAX_NUM_SEQS=128            # Max concurrent sequences
VOXCPM_MAX_MODEL_LEN=2048          # Max sequence length
```

### Memory Usage

| Configuration | Memory/GPU | Max Concurrency | Throughput |
|--------------|------------|-----------------|------------|
| util=0.4, seqs=128 | 7.4 GB | 128 | 166x |
| util=0.5, seqs=128 | 10.7 GB | 128 | 166x |
| util=0.7, seqs=128 | 14.8 GB | 128 | 164x |
| util=0.9, seqs=128 | 19.2 GB | 128 | 170x |

**Key findings:**
- Minimum `util` is 0.4 (lower values cause KV Cache allocation failure)
- `seqs` has minimal impact on memory (~400MB from 8 to 128)
- Memory is primarily determined by `util`

## Performance Benchmark

### Test Results (VoxCPM 1.5, RTX 4090)

| Deployment | GPUs | seqs | Peak Throughput | Memory/GPU |
|------------|------|------|-----------------|------------|
| 2×2 Data Parallel | 4 | 128 | 166x | 7.4 GB |
| 1×4 Tensor Parallel | 4 | 128 | **248x** | 7.5 GB |

**Throughput = Total audio duration / Total processing time**
- 248x means generating 248 seconds of audio per second

### Run Benchmarks

```bash
cd benchmark
python benchmark_limit.py
```

See [benchmark/README.md](benchmark/README.md) for detailed benchmark documentation.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VOXCPM_MODEL_PATH` | Auto-detect | Path to VoxCPM model |
| `VOXCPM_DEVICES` | All GPUs | GPU device indices (e.g., "0,1") |
| `VOXCPM_GPU_MEMORY_UTILIZATION` | 0.9 | GPU memory utilization (0.4-0.9) |
| `VOXCPM_MAX_NUM_SEQS` | 512 | Max concurrent sequences |
| `VOXCPM_MAX_MODEL_LEN` | 4096 | Max sequence length |

## Project Structure

```
nanovllm-voxcpm/
├── Dockerfile           # Docker image definition
├── docker-compose.yml   # Docker Compose configuration
├── api_server/          # OpenAI compatible API server
│   └── app.py
├── benchmark/           # Performance benchmark tools
│   ├── README.md
│   ├── benchmark.py
│   ├── benchmark_fullload.py
│   ├── benchmark_limit.py
│   └── voxcpm_benchmark_report.md
├── fastapi/             # Legacy FastAPI example
├── nanovllm_voxcpm/     # Core inference engine
├── example.py           # Basic usage example
└── example_sync.py      # Synchronous example
```

## Acknowledgments

- [VoxCPM](https://github.com/OpenBMB/VoxCPM)
- [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)

## License

MIT License

## Known Issue

If you see the error below:
```
ValueError: Missing parameters: ['base_lm.embed_tokens.weight', ...]
```

It's because nanovllm reads model parameters from `.safetensors` files, but the original format of VoxCPM is `.pt`. You can download the [safetensor](https://huggingface.co/euphoricpenguin22/VoxCPM-0.5B-Safetensors/blob/main/model.safetensors) file manually and put it into the model folder.
