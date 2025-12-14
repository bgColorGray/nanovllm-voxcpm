# VoxCPM 1.5 Benchmark

VoxCPM 1.5 显存优化与性能测试工具集。

## 测试结论

### 推荐配置

```bash
VOXCPM_GPU_MEMORY_UTILIZATION=0.4
VOXCPM_MAX_NUM_SEQS=128
VOXCPM_MAX_MODEL_LEN=2048

# 性能指标 (2卡数据并行):
# - 显存: ~7.4 GB/卡
# - 最大并发: 128
# - 满载吞吐: 157x
```

### 关键发现

| 发现 | 说明 |
|------|------|
| util 最低 0.4 | 低于 0.4 会导致 KV Cache 不足 |
| seqs 对显存影响小 | 从 8 到 128 仅增加 ~400MB |
| 超并发会排队 | 超过 seqs 的请求不会失败，只是延迟增加 |
| 张量并行更高效 | 1×4 卡吞吐量 248x，比 2×2 卡的 166x 高 50% |

### 部署方式对比

| 部署方式 | 卡数 | seqs | 峰值吞吐 | 显存/卡 |
|----------|------|------|----------|---------|
| 2×2 数据并行 | 4 | 128 | 166x | 7.4 GB |
| 1×4 张量并行 | 4 | 128 | 248x | 7.5 GB |

## 测试脚本

| 脚本 | 用途 |
|------|------|
| `benchmark.py` | 基础参数组合测试 |
| `benchmark_fullload.py` | 满负载吞吐量测试 |
| `benchmark_limit.py` | 极限并发探索测试 |

## 使用方法

### 1. 设置环境变量

```bash
export VOXCPM_MODEL_PATH=/path/to/VoxCPM1.5
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VOXCPM_DEVICES=0,1,2,3
export VOXCPM_GPU_MEMORY_UTILIZATION=0.4
export VOXCPM_MAX_NUM_SEQS=128
export VOXCPM_MAX_MODEL_LEN=2048
```

### 2. 运行测试

```bash
cd benchmark
python benchmark_limit.py
```

### 3. 查看结果

测试结果保存在 `results/` 目录下，包含：
- `results.json`: 原始数据
- `report.md`: Markdown 报告

## 参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `VOXCPM_GPU_MEMORY_UTILIZATION` | GPU 显存利用率上限 | 0.4 |
| `VOXCPM_MAX_NUM_SEQS` | 最大并发序列数 | 128 |
| `VOXCPM_MAX_MODEL_LEN` | 最大序列长度 | 2048 |

### max_model_len 选择

| 业务场景 | 文本长度 | 推荐值 |
|----------|----------|--------|
| 短句 TTS | < 50 字 | 512 |
| 段落 TTS | < 200 字 | 1024 |
| 长文章 TTS | < 500 字 | 2048 |

## 测试环境

- GPU: NVIDIA RTX 4090 (24GB)
- 模型: VoxCPM 1.5 (44.1kHz)
- Python: 3.11
- 框架: nanovllm-voxcpm

## 详细报告

完整测试报告见 [voxcpm_benchmark_report.md](voxcpm_benchmark_report.md)
