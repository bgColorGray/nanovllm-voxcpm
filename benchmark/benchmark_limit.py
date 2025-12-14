#!/usr/bin/env python3
"""VoxCPM 1.5 极限并发测试
- 探索每个 util 的最大 max_num_seqs
- 递增 seqs 直到启动失败（OOM）
"""

import asyncio
import time
import json
import subprocess
import os
import signal
import base64
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional
from datetime import datetime
import aiohttp

# ==================== 配置 ====================

@dataclass
class TestConfig:
    gpu_memory_utilization: float
    max_num_seqs: int
    max_model_len: int = 2048

@dataclass
class LimitResult:
    util: float
    seqs: int
    started: bool
    idle_memory_mb: float = 0.0
    fullload_throughput: float = 0.0
    fullload_rtf: float = 0.0
    fullload_success_rate: float = 0.0
    error: str = ""

# GPU 分组配置
GPU_GROUPS = [
    {"id": "group0", "devices": [3, 4], "port": 8081},
    {"id": "group1", "devices": [5, 6], "port": 8082},
]

# 测试参数
UTIL_LEVELS = [0.5, 0.7, 0.9]
SEQS_CANDIDATES = [8, 16, 32, 64, 128]

# 参考音频配置
REFERENCE_AUDIO_PATH = "/home/estar/voxcpm-docker/voices/reference/reference_audio.wav"
REFERENCE_TEXT_PATH = "/home/estar/voxcpm-docker/voices/reference/reference_text.txt"
VOICE_NAME = "benchmark_voice"

# 测试文本
TEST_TEXT = "人工智能技术正在快速发展，它已经渗透到我们生活的方方面面。"

# ==================== 服务管理 ====================

class ServiceGroup:
    def __init__(self, group_config: dict, model_path: str, api_server_dir: str):
        self.group_id = group_config["id"]
        self.devices = group_config["devices"]
        self.port = group_config["port"]
        self.model_path = model_path
        self.api_server_dir = api_server_dir
        self.process = None

    def start(self, config: TestConfig) -> bool:
        env = os.environ.copy()
        env["VOXCPM_MODEL_PATH"] = self.model_path
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.devices))
        env["VOXCPM_DEVICES"] = ",".join(map(str, range(len(self.devices))))
        env["VOXCPM_GPU_MEMORY_UTILIZATION"] = str(config.gpu_memory_utilization)
        env["VOXCPM_MAX_NUM_SEQS"] = str(config.max_num_seqs)
        env["VOXCPM_MAX_MODEL_LEN"] = str(config.max_model_len)

        cmd = ["/home/estar/miniconda3/envs/voxcpm/bin/python", "-m", "uvicorn", "api_server.app:app",
               "--host", "0.0.0.0", "--port", str(self.port)]

        log_file = open(f"/tmp/voxcpm_{self.group_id}.log", "w")
        self.process = subprocess.Popen(cmd, cwd=self.api_server_dir, env=env,
                                        stdout=log_file, stderr=subprocess.STDOUT,
                                        preexec_fn=os.setsid)
        print(f"  [{self.group_id}] PID={self.process.pid}, GPU={self.devices}, Port={self.port}")
        return True

    def stop(self):
        if self.process:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=30)
            except: pass
            self.process = None

    async def wait_ready(self, timeout: int = 300) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"http://localhost:{self.port}/health",
                                          timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            return True
            except: pass
            await asyncio.sleep(2)
        return False

class ServiceManager:
    def __init__(self, model_path: str = "/home/estar/voxcpm-docker/VoxCPM1.5",
                 api_server_dir: str = "/home/estar/voxcpm-docker/nanovllm-voxcpm"):
        self.groups = [ServiceGroup(g, model_path, api_server_dir) for g in GPU_GROUPS]

    def start_all(self, config: TestConfig):
        print(f"\n启动服务 (util={config.gpu_memory_utilization}, seqs={config.max_num_seqs})")
        for group in self.groups:
            group.start(config)

    def stop_all(self):
        print("停止所有服务...")
        for group in self.groups:
            group.stop()
        subprocess.run("pkill -9 -f 'uvicorn.*app:app'", shell=True, capture_output=True)
        subprocess.run("pkill -9 -f 'multiprocessing.spawn'", shell=True, capture_output=True)
        time.sleep(5)

    async def wait_all_ready(self, timeout: int = 300) -> bool:
        print("等待服务就绪...")
        results = await asyncio.gather(*[g.wait_ready(timeout) for g in self.groups])
        ready_count = sum(results)
        print(f"  {ready_count}/{len(self.groups)} 组服务就绪")
        return all(results)

# ==================== 测试执行 ====================

class LimitBenchmark:
    def __init__(self):
        self.ports = [g["port"] for g in GPU_GROUPS]

    def get_avg_gpu_memory(self) -> float:
        """获取所有 GPU 的平均显存"""
        memories = []
        for group in GPU_GROUPS:
            for device_id in group["devices"]:
                proc = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits", f"--id={device_id}"],
                    capture_output=True, text=True)
                try:
                    memories.append(float(proc.stdout.strip()))
                except:
                    pass
        return sum(memories) / len(memories) if memories else 0

    async def create_voice(self, port: int) -> bool:
        """创建克隆音色"""
        with open(REFERENCE_AUDIO_PATH, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode()
        with open(REFERENCE_TEXT_PATH, "r") as f:
            prompt_text = f.read().strip()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"http://localhost:{port}/v1/voices",
                    json={"voice_name": VOICE_NAME, "prompt_wav_base64": audio_base64,
                          "prompt_text": prompt_text, "replace": True},
                    timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    return resp.status == 200
        except Exception as e:
            print(f"  创建音色失败: {e}")
            return False

    async def single_request(self, port: int, text: str) -> dict:
        """单个请求"""
        start = time.perf_counter()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"http://localhost:{port}/v1/audio/speech",
                    json={"input": text, "model": "voxcpm", "voice": VOICE_NAME},
                    timeout=aiohttp.ClientTimeout(total=300)) as response:
                    audio_data = await response.read()

            gen_time = time.perf_counter() - start
            audio_duration = (len(audio_data) - 44) / (44100 * 2)
            rtf = gen_time / audio_duration if audio_duration > 0 else -1

            return {"success": True, "gen_time": gen_time, "audio_duration": audio_duration, "rtf": rtf}
        except Exception as e:
            return {"success": False, "gen_time": time.perf_counter() - start, "audio_duration": 0, "rtf": -1, "error": str(e)}

    async def run_fullload_test(self, port: int, n_concurrent: int) -> dict:
        """满载测试"""
        print(f"  满载测试: {n_concurrent} 并发...")

        tasks = [self.single_request(port, TEST_TEXT) for _ in range(n_concurrent)]
        start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start

        success_results = [r for r in results if r["success"]]
        success_count = len(success_results)

        if success_count > 0:
            avg_rtf = sum(r["rtf"] for r in success_results) / success_count
            total_audio = sum(r["audio_duration"] for r in success_results)
            throughput = total_audio / total_time
        else:
            avg_rtf = -1
            throughput = 0

        return {
            "success_rate": success_count / n_concurrent,
            "avg_rtf": avg_rtf,
            "throughput": throughput,
        }

    async def test_config(self, config: TestConfig, service: ServiceManager) -> LimitResult:
        """测试单个配置"""
        result = LimitResult(
            util=config.gpu_memory_utilization,
            seqs=config.max_num_seqs,
            started=False
        )

        # 停止旧服务
        service.stop_all()

        # 启动服务
        service.start_all(config)

        # 等待就绪
        if not await service.wait_all_ready(timeout=300):
            result.started = False
            result.error = "启动超时或OOM"
            print(f"  ❌ 启动失败")
            service.stop_all()
            return result

        result.started = True
        print(f"  ✓ 启动成功")

        # 测量空载显存
        await asyncio.sleep(3)
        result.idle_memory_mb = self.get_avg_gpu_memory()
        print(f"  空载显存: {result.idle_memory_mb:.0f} MB/卡")

        # 创建克隆音色
        if not await self.create_voice(self.ports[0]):
            result.error = "创建音色失败"
            service.stop_all()
            return result

        # 满载测试
        fullload = await self.run_fullload_test(self.ports[0], config.max_num_seqs)
        result.fullload_success_rate = fullload["success_rate"]
        result.fullload_rtf = fullload["avg_rtf"]
        result.fullload_throughput = fullload["throughput"]

        print(f"  满载结果: 成功率={result.fullload_success_rate*100:.0f}%, "
              f"RTF={result.fullload_rtf:.3f}, 吞吐量={result.fullload_throughput:.1f}x")

        service.stop_all()
        return result

# ==================== 主流程 ====================

def save_results(results: List[LimitResult], output_dir: str):
    """保存结果"""
    data = [asdict(r) for r in results]
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def generate_report(results: List[LimitResult], output_dir: str):
    """生成报告"""
    lines = [
        "# VoxCPM 1.5 极限并发测试报告\n",
        f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
    ]

    # 按 util 分组
    for util in UTIL_LEVELS:
        util_results = [r for r in results if r.util == util]
        if not util_results:
            continue

        lines.append(f"\n## util = {util}\n")
        lines.append("| seqs | 启动 | 显存(MB) | 满载吞吐 | RTF | 成功率 |")
        lines.append("|------|------|----------|----------|-----|--------|")

        max_working = None
        for r in util_results:
            if r.started:
                max_working = r
                lines.append(f"| {r.seqs} | ✓ | {r.idle_memory_mb:.0f} | "
                            f"{r.fullload_throughput:.1f}x | {r.fullload_rtf:.3f} | "
                            f"{r.fullload_success_rate*100:.0f}% |")
            else:
                lines.append(f"| {r.seqs} | ❌ | - | - | - | {r.error} |")

        if max_working:
            lines.append(f"\n**util={util} 最大支持**: seqs={max_working.seqs}, "
                        f"满载吞吐={max_working.fullload_throughput:.1f}x\n")

    # 总结
    lines.append("\n## 结论\n")
    for util in UTIL_LEVELS:
        util_results = [r for r in results if r.util == util and r.started]
        if util_results:
            best = max(util_results, key=lambda x: x.seqs)
            lines.append(f"- **util={util}**: 最大 seqs={best.seqs}, "
                        f"显存={best.idle_memory_mb:.0f}MB, 满载吞吐={best.fullload_throughput:.1f}x")

    with open(f"{output_dir}/report.md", "w") as f:
        f.write("\n".join(lines))

async def main():
    output_dir = f"./results/limit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("VoxCPM 1.5 极限并发测试")
    print("目标: 找出每个 util 的最大 max_num_seqs")
    print("="*60)

    service = ServiceManager()
    benchmark = LimitBenchmark()
    results = []

    for util in UTIL_LEVELS:
        print(f"\n\n{'#'*60}")
        print(f"# 测试 util = {util}")
        print('#'*60)

        for seqs in SEQS_CANDIDATES:
            print(f"\n--- 尝试 seqs = {seqs} ---")

            config = TestConfig(util, seqs)
            result = await benchmark.test_config(config, service)
            results.append(result)

            # 保存中间结果
            save_results(results, output_dir)

            # 如果启动失败，跳过更大的 seqs
            if not result.started:
                print(f"  跳过更大的 seqs（已达极限）")
                break

    # 生成最终报告
    generate_report(results, output_dir)
    print(f"\n\n测试完成! 结果保存在: {output_dir}")

if __name__ == "__main__":
    asyncio.run(main())
