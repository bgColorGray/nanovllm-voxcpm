#!/usr/bin/env python3
"""VoxCPM 1.5 满负载并发测试
- 递增并发测试直到 max_num_seqs
- 测量：显存、RTF、吞吐量、成功率
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
class ConcurrentResult:
    concurrent: int
    success_count: int
    total_count: int
    success_rate: float
    avg_rtf: float
    throughput: float  # 音频秒/实际秒
    total_time: float
    gpu_memory_mb: float

@dataclass
class TestResult:
    config: TestConfig
    idle_memory_mb: float = 0.0
    concurrent_results: List[ConcurrentResult] = field(default_factory=list)
    success: bool = True
    error: str = ""

# GPU 分组配置
GPU_GROUPS = [
    {"id": "group0", "devices": [3, 4], "port": 8081},
    {"id": "group1", "devices": [5, 6], "port": 8082},
]

# 3 个代表性配置
TEST_CONFIGS = [
    TestConfig(0.5, 8, 2048),   # 低配：测试 1,2,4,8 并发
    TestConfig(0.7, 16, 2048),  # 中配：测试 1,2,4,8,16 并发
    TestConfig(0.9, 32, 2048),  # 高配：测试 1,2,4,8,16,32 并发
]

# 参考音频配置
REFERENCE_AUDIO_PATH = "/home/estar/voxcpm-docker/voices/reference/reference_audio.wav"
REFERENCE_TEXT_PATH = "/home/estar/voxcpm-docker/voices/reference/reference_text.txt"
VOICE_NAME = "benchmark_voice"

# 测试文本（使用短文本减少单次测试时间）
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
        subprocess.run("pkill -f 'uvicorn.*app:app'", shell=True, capture_output=True)
        time.sleep(3)

    async def wait_all_ready(self, timeout: int = 300) -> bool:
        print("等待服务就绪...")
        results = await asyncio.gather(*[g.wait_ready(timeout) for g in self.groups])
        print(f"  {sum(results)}/{len(self.groups)} 组服务就绪")
        return all(results)

# ==================== 测试执行 ====================

class FullLoadBenchmark:
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
                memories.append(float(proc.stdout.strip()))
        return sum(memories) / len(memories) if memories else 0

    async def create_voice_on_all(self) -> bool:
        """在所有服务上创建克隆音色"""
        print(f"创建克隆音色...")
        with open(REFERENCE_AUDIO_PATH, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode()
        with open(REFERENCE_TEXT_PATH, "r") as f:
            prompt_text = f.read().strip()

        async def create_on_port(port: int):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"http://localhost:{port}/v1/voices",
                        json={"voice_name": VOICE_NAME, "prompt_wav_base64": audio_base64,
                              "prompt_text": prompt_text, "replace": True},
                        timeout=aiohttp.ClientTimeout(total=60)) as resp:
                        return resp.status == 200
            except Exception as e:
                print(f"  Port {port} 创建异常: {e}")
                return False

        results = await asyncio.gather(*[create_on_port(p) for p in self.ports])
        success = sum(results)
        print(f"  {success}/{len(self.ports)} 组创建成功")
        return success > 0  # 至少一组成功

    async def single_request(self, port: int, text: str) -> dict:
        """单个请求，返回结果详情"""
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

            return {
                "success": True,
                "gen_time": gen_time,
                "audio_duration": audio_duration,
                "rtf": rtf
            }
        except Exception as e:
            return {
                "success": False,
                "gen_time": time.perf_counter() - start,
                "audio_duration": 0,
                "rtf": -1,
                "error": str(e)
            }

    async def measure_concurrent(self, port: int, n_concurrent: int, text: str) -> ConcurrentResult:
        """测量指定并发数的性能"""
        # 并发发送请求
        tasks = [self.single_request(port, text) for _ in range(n_concurrent)]
        start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start

        # 统计
        success_results = [r for r in results if r["success"]]
        success_count = len(success_results)

        if success_count > 0:
            avg_rtf = sum(r["rtf"] for r in success_results) / success_count
            total_audio = sum(r["audio_duration"] for r in success_results)
            throughput = total_audio / total_time
        else:
            avg_rtf = -1
            throughput = 0

        # 测量当前显存
        gpu_memory = self.get_avg_gpu_memory()

        return ConcurrentResult(
            concurrent=n_concurrent,
            success_count=success_count,
            total_count=n_concurrent,
            success_rate=success_count / n_concurrent,
            avg_rtf=avg_rtf,
            throughput=throughput,
            total_time=total_time,
            gpu_memory_mb=gpu_memory
        )

    async def run_test(self, config: TestConfig) -> TestResult:
        """运行单个配置的完整测试"""
        result = TestResult(config=config)
        print(f"\n{'='*60}")
        print(f"满负载测试: util={config.gpu_memory_utilization}, seqs={config.max_num_seqs}")
        print('='*60)

        try:
            await asyncio.sleep(5)

            # 1. 测量空载显存
            result.idle_memory_mb = self.get_avg_gpu_memory()
            print(f"空载显存: {result.idle_memory_mb:.0f} MB/卡")

            # 2. 创建克隆音色
            if not await self.create_voice_on_all():
                result.success = False
                result.error = "创建克隆音色失败"
                return result

            # 3. 递增并发测试
            # 生成并发数序列: 1, 2, 4, 8, ... 直到 max_num_seqs
            concurrent_levels = []
            n = 1
            while n <= config.max_num_seqs:
                concurrent_levels.append(n)
                n *= 2
            if concurrent_levels[-1] != config.max_num_seqs:
                concurrent_levels.append(config.max_num_seqs)

            print(f"\n测试并发级别: {concurrent_levels}")
            print("-" * 60)
            print(f"{'并发':>4} | {'成功率':>6} | {'RTF':>6} | {'吞吐量':>8} | {'显存(MB)':>10} | {'耗时(s)':>8}")
            print("-" * 60)

            # 使用第一个端口进行测试
            test_port = self.ports[0]

            for n_concurrent in concurrent_levels:
                # 每个并发级别测试 2 轮取平均
                round_results = []
                for _ in range(2):
                    cr = await self.measure_concurrent(test_port, n_concurrent, TEST_TEXT)
                    round_results.append(cr)
                    await asyncio.sleep(1)  # 短暂休息

                # 计算平均值
                avg_result = ConcurrentResult(
                    concurrent=n_concurrent,
                    success_count=int(sum(r.success_count for r in round_results) / len(round_results)),
                    total_count=n_concurrent,
                    success_rate=sum(r.success_rate for r in round_results) / len(round_results),
                    avg_rtf=sum(r.avg_rtf for r in round_results) / len(round_results),
                    throughput=sum(r.throughput for r in round_results) / len(round_results),
                    total_time=sum(r.total_time for r in round_results) / len(round_results),
                    gpu_memory_mb=sum(r.gpu_memory_mb for r in round_results) / len(round_results)
                )

                result.concurrent_results.append(avg_result)

                print(f"{avg_result.concurrent:>4} | {avg_result.success_rate*100:>5.1f}% | "
                      f"{avg_result.avg_rtf:>6.3f} | {avg_result.throughput:>7.2f}x | "
                      f"{avg_result.gpu_memory_mb:>10.0f} | {avg_result.total_time:>8.2f}")

            result.success = True

        except Exception as e:
            result.success = False
            result.error = str(e)
            print(f"测试异常: {e}")

        return result

# ==================== 主流程 ====================

def save_results(results: List[TestResult], output_dir: str):
    """保存结果为 JSON"""
    data = []
    for r in results:
        data.append({
            "config": asdict(r.config),
            "idle_memory_mb": r.idle_memory_mb,
            "concurrent_results": [asdict(cr) for cr in r.concurrent_results],
            "success": r.success,
            "error": r.error,
        })
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def generate_report(results: List[TestResult], output_dir: str):
    """生成 Markdown 报告"""
    lines = [
        "# VoxCPM 1.5 满负载并发测试报告\n",
        f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "配置: 4卡分2组, 每组2卡, 使用克隆音色\n",
    ]

    for r in results:
        if not r.success:
            lines.append(f"\n## 配置 util={r.config.gpu_memory_utilization}, seqs={r.config.max_num_seqs}\n")
            lines.append(f"**失败**: {r.error}\n")
            continue

        lines.append(f"\n## 配置 util={r.config.gpu_memory_utilization}, seqs={r.config.max_num_seqs}\n")
        lines.append(f"空载显存: {r.idle_memory_mb:.0f} MB/卡\n")
        lines.append("\n| 并发 | 成功率 | RTF | 吞吐量 | 显存(MB) | 耗时(s) |")
        lines.append("|------|--------|-----|--------|----------|---------|")

        for cr in r.concurrent_results:
            lines.append(f"| {cr.concurrent} | {cr.success_rate*100:.1f}% | {cr.avg_rtf:.3f} | "
                        f"{cr.throughput:.2f}x | {cr.gpu_memory_mb:.0f} | {cr.total_time:.2f} |")

    # 汇总分析
    lines.append("\n## 结论\n")
    for r in results:
        if r.success and r.concurrent_results:
            best = max(r.concurrent_results, key=lambda x: x.throughput if x.success_rate >= 0.9 else 0)
            lines.append(f"- **util={r.config.gpu_memory_utilization}, seqs={r.config.max_num_seqs}**: "
                        f"最佳并发={best.concurrent}, 吞吐量={best.throughput:.2f}x, "
                        f"显存={best.gpu_memory_mb:.0f}MB")

    with open(f"{output_dir}/report.md", "w") as f:
        f.write("\n".join(lines))

async def main():
    output_dir = f"./results/fullload_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("VoxCPM 1.5 满负载并发测试")
    print("配置: 4卡分2组, 每组2卡, 使用克隆音色")
    print("="*60)

    service = ServiceManager()
    benchmark = FullLoadBenchmark()
    results = []

    for i, config in enumerate(TEST_CONFIGS):
        print(f"\n\n{'#'*60}")
        print(f"# 配置 {i+1}/{len(TEST_CONFIGS)}: util={config.gpu_memory_utilization}, seqs={config.max_num_seqs}")
        print('#'*60)

        service.stop_all()
        service.start_all(config)

        if not await service.wait_all_ready(timeout=300):
            print("服务启动超时!")
            results.append(TestResult(config=config, success=False, error="启动超时"))
            continue

        result = await benchmark.run_test(config)
        results.append(result)
        save_results(results, output_dir)

    service.stop_all()
    generate_report(results, output_dir)
    print(f"\n测试完成! 结果保存在: {output_dir}")

if __name__ == "__main__":
    asyncio.run(main())
