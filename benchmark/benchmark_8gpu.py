#!/usr/bin/env python3
"""VoxCPM 1.5 八卡压力测试
- 1×8 卡张量并行部署
- 测试不同并发级别的性能
"""

import asyncio
import time
import json
import subprocess
import os
import signal
import base64
from dataclasses import dataclass, asdict
from typing import List, Dict
from datetime import datetime
import aiohttp

# ==================== 配置 ====================

# 4 卡配置 (使用占用最少的 GPU)
GPU_DEVICES = [3, 4, 5, 6]
PORT = 8081
MODEL_PATH = "/home/estar/voxcpm-docker/VoxCPM1.5"
API_SERVER_DIR = "/home/estar/voxcpm-docker/nanovllm-voxcpm"

# 测试参数
UTIL = 0.4
MAX_NUM_SEQS = 128
MAX_MODEL_LEN = 2048

# 参考音频配置
REFERENCE_AUDIO_PATH = "/home/estar/voxcpm-docker/voices/reference/reference_audio.wav"
REFERENCE_TEXT_PATH = "/home/estar/voxcpm-docker/voices/reference/reference_text.txt"
VOICE_NAME = "benchmark_voice"

# 测试文本 (~100字)
TEST_TEXT = """人工智能技术正在以前所未有的速度改变着我们的世界。从智能语音助手到自动驾驶汽车，从医疗诊断到金融分析，AI已经深入到各行各业。语音合成技术作为人工智能的重要分支，让机器能够以自然流畅的方式与人类交流。"""

# 并发测试级别
CONCURRENCY_LEVELS = [32, 64, 128, 192, 256, 384, 512]

# ==================== 数据结构 ====================

@dataclass
class TestResult:
    concurrency: int
    success_count: int
    fail_count: int
    wall_time: float
    throughput: float  # 音频秒/实际秒
    ttfb_min: float
    ttfb_avg: float
    ttfb_max: float
    avg_audio_duration: float
    avg_gen_time: float

# ==================== 服务管理 ====================

def stop_service():
    """停止现有服务"""
    subprocess.run("pkill -9 -f 'uvicorn.*app:app'", shell=True, capture_output=True)
    subprocess.run("pkill -9 -f 'multiprocessing.spawn'", shell=True, capture_output=True)
    time.sleep(5)

def start_service(seqs: int = MAX_NUM_SEQS) -> subprocess.Popen:
    """启动 8 卡服务"""
    env = os.environ.copy()
    env["VOXCPM_MODEL_PATH"] = MODEL_PATH
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, GPU_DEVICES))
    env["VOXCPM_DEVICES"] = ",".join(map(str, range(len(GPU_DEVICES))))
    env["VOXCPM_GPU_MEMORY_UTILIZATION"] = str(UTIL)
    env["VOXCPM_MAX_NUM_SEQS"] = str(seqs)
    env["VOXCPM_MAX_MODEL_LEN"] = str(MAX_MODEL_LEN)

    cmd = ["/home/estar/miniconda3/envs/voxcpm/bin/python", "-m", "uvicorn", "api_server.app:app",
           "--host", "0.0.0.0", "--port", str(PORT)]

    log_file = open(f"/tmp/voxcpm_8gpu.log", "w")
    process = subprocess.Popen(cmd, cwd=API_SERVER_DIR, env=env,
                               stdout=log_file, stderr=subprocess.STDOUT,
                               preexec_fn=os.setsid)
    print(f"  启动服务 PID={process.pid}, GPU={GPU_DEVICES}, seqs={seqs}")
    return process

async def wait_ready(timeout: int = 300) -> bool:
    """等待服务就绪"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://localhost:{PORT}/health",
                                      timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        return True
        except:
            pass
        await asyncio.sleep(2)
    return False

def get_gpu_memory() -> Dict[int, float]:
    """获取各 GPU 显存"""
    result = {}
    for device_id in GPU_DEVICES:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits", f"--id={device_id}"],
            capture_output=True, text=True)
        try:
            result[device_id] = float(proc.stdout.strip())
        except:
            result[device_id] = 0
    return result

# ==================== 测试执行 ====================

async def create_voice() -> bool:
    """创建克隆音色"""
    with open(REFERENCE_AUDIO_PATH, "rb") as f:
        audio_base64 = base64.b64encode(f.read()).decode()
    with open(REFERENCE_TEXT_PATH, "r") as f:
        prompt_text = f.read().strip()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"http://localhost:{PORT}/v1/voices",
                json={"voice_name": VOICE_NAME, "prompt_wav_base64": audio_base64,
                      "prompt_text": prompt_text, "replace": True},
                timeout=aiohttp.ClientTimeout(total=60)) as resp:
                return resp.status == 200
    except Exception as e:
        print(f"  创建音色失败: {e}")
        return False

async def single_request_with_ttfb(req_id: int) -> dict:
    """单个请求，返回 TTFB 和完整结果"""
    start = time.perf_counter()
    ttfb = -1
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"http://localhost:{PORT}/v1/audio/speech/stream",
                json={"input": TEST_TEXT, "model": "voxcpm", "voice": VOICE_NAME},
                timeout=aiohttp.ClientTimeout(total=300)) as response:

                chunks = []
                async for chunk in response.content.iter_chunked(4096):
                    if ttfb < 0:
                        ttfb = (time.perf_counter() - start) * 1000
                    chunks.append(chunk)

                audio_data = b''.join(chunks)

        gen_time = time.perf_counter() - start
        # PCM 16bit mono 44100Hz
        audio_duration = len(audio_data) / (44100 * 2)

        return {
            "success": True,
            "ttfb": ttfb,
            "gen_time": gen_time,
            "audio_duration": audio_duration,
        }
    except Exception as e:
        return {
            "success": False,
            "ttfb": -1,
            "gen_time": time.perf_counter() - start,
            "audio_duration": 0,
            "error": str(e)
        }

async def run_concurrent_test(n_concurrent: int) -> TestResult:
    """运行并发测试"""
    print(f"\n  测试 {n_concurrent} 并发...")

    tasks = [single_request_with_ttfb(i) for i in range(n_concurrent)]
    start = time.perf_counter()
    results = await asyncio.gather(*tasks)
    wall_time = time.perf_counter() - start

    success_results = [r for r in results if r["success"]]
    fail_results = [r for r in results if not r["success"]]

    if success_results:
        ttfbs = [r["ttfb"] for r in success_results]
        total_audio = sum(r["audio_duration"] for r in success_results)
        throughput = total_audio / wall_time

        return TestResult(
            concurrency=n_concurrent,
            success_count=len(success_results),
            fail_count=len(fail_results),
            wall_time=wall_time,
            throughput=throughput,
            ttfb_min=min(ttfbs),
            ttfb_avg=sum(ttfbs) / len(ttfbs),
            ttfb_max=max(ttfbs),
            avg_audio_duration=sum(r["audio_duration"] for r in success_results) / len(success_results),
            avg_gen_time=sum(r["gen_time"] for r in success_results) / len(success_results),
        )
    else:
        return TestResult(
            concurrency=n_concurrent,
            success_count=0,
            fail_count=len(fail_results),
            wall_time=wall_time,
            throughput=0,
            ttfb_min=-1,
            ttfb_avg=-1,
            ttfb_max=-1,
            avg_audio_duration=0,
            avg_gen_time=0,
        )

# ==================== 主流程 ====================

async def main():
    output_dir = f"./results/8gpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("VoxCPM 1.5 八卡压力测试")
    print(f"配置: 1×8 卡, util={UTIL}, seqs={MAX_NUM_SEQS}, len={MAX_MODEL_LEN}")
    print(f"测试文本: {len(TEST_TEXT)} 字")
    print("="*60)

    # 停止旧服务
    print("\n停止现有服务...")
    stop_service()

    # 启动 8 卡服务
    print("\n启动 8 卡服务...")
    process = start_service()

    # 等待就绪
    print("等待服务就绪...")
    if not await wait_ready(timeout=300):
        print("❌ 服务启动失败!")
        stop_service()
        return

    print("✓ 服务就绪!")

    # 记录显存
    await asyncio.sleep(3)
    gpu_memory = get_gpu_memory()
    print(f"\n显存占用:")
    for gpu_id, mem in gpu_memory.items():
        print(f"  GPU {gpu_id}: {mem:.0f} MB")
    avg_memory = sum(gpu_memory.values()) / len(gpu_memory)
    print(f"  平均: {avg_memory:.0f} MB/卡")

    # 创建音色
    print("\n创建克隆音色...")
    if not await create_voice():
        print("❌ 创建音色失败!")
        stop_service()
        return
    print("✓ 音色创建成功!")

    # 运行并发测试
    print("\n" + "="*60)
    print("开始并发测试")
    print("="*60)

    results = []
    for n in CONCURRENCY_LEVELS:
        result = await run_concurrent_test(n)
        results.append(result)

        print(f"    成功: {result.success_count}/{n}")
        print(f"    吞吐量: {result.throughput:.1f}x")
        print(f"    TTFB: min={result.ttfb_min:.0f}ms, avg={result.ttfb_avg:.0f}ms, max={result.ttfb_max:.0f}ms")
        print(f"    Wall Time: {result.wall_time:.1f}s")

        # 如果失败率太高，停止测试
        if result.fail_count > n * 0.5:
            print(f"  ⚠️ 失败率过高，停止更高并发测试")
            break

        await asyncio.sleep(3)

    # 保存结果
    data = [asdict(r) for r in results]
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # 生成报告
    print("\n\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    print(f"配置: 8 卡, util={UTIL}, seqs={MAX_NUM_SEQS}")
    print(f"平均显存: {avg_memory:.0f} MB/卡")
    print(f"\n{'并发':>6} | {'成功率':>8} | {'吞吐量':>8} | {'TTFB avg':>10} | {'TTFB max':>10} | {'Wall Time':>10}")
    print("-"*70)
    for r in results:
        rate = f"{r.success_count}/{r.concurrency}"
        print(f"{r.concurrency:>6} | {rate:>8} | {r.throughput:>7.1f}x | {r.ttfb_avg:>9.0f}ms | {r.ttfb_max:>9.0f}ms | {r.wall_time:>9.1f}s")

    # 生成 markdown 报告
    lines = [
        f"# VoxCPM 1.5 八卡压力测试报告\n",
        f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"\n## 配置\n",
        f"| 项目 | 值 |",
        f"|------|-----|",
        f"| GPU | 8× RTX 4090 |",
        f"| util | {UTIL} |",
        f"| seqs | {MAX_NUM_SEQS} |",
        f"| len | {MAX_MODEL_LEN} |",
        f"| 平均显存 | {avg_memory:.0f} MB/卡 |",
        f"| 测试文本 | {len(TEST_TEXT)} 字 |",
        f"\n## 测试结果\n",
        f"| 并发 | 成功率 | 吞吐量 | TTFB avg | TTFB max | Wall Time |",
        f"|------|--------|--------|----------|----------|-----------|",
    ]
    for r in results:
        rate = f"{r.success_count}/{r.concurrency}"
        lines.append(f"| {r.concurrency} | {rate} | {r.throughput:.1f}x | {r.ttfb_avg:.0f}ms | {r.ttfb_max:.0f}ms | {r.wall_time:.1f}s |")

    with open(f"{output_dir}/report.md", "w") as f:
        f.write("\n".join(lines))

    print(f"\n结果保存在: {output_dir}")

    # 停止服务
    stop_service()

if __name__ == "__main__":
    asyncio.run(main())
