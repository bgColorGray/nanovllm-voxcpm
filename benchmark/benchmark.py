#!/usr/bin/env python3
"""VoxCPM 1.5 显存参数优化基准测试
- 4卡分2组并发测试
- 使用统一克隆音色
"""

import asyncio
import time
import json
import subprocess
import os
import signal
import sys
import base64
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict
from datetime import datetime
import aiohttp

# ==================== 配置 ====================

@dataclass
class TestConfig:
    gpu_memory_utilization: float
    max_num_seqs: int
    max_model_len: int

@dataclass  
class TestResult:
    config: TestConfig
    gpu_memory_mb: Dict[str, List[float]] = field(default_factory=dict)
    avg_memory_per_gpu: float = 0.0
    ttfb_ms: float = 0.0
    rtf: float = 0.0
    generation_time: float = 0.0
    audio_duration: float = 0.0
    concurrent_4group_success: int = 0
    concurrent_4group_total_time: float = 0.0
    success: bool = True
    error: str = ""

# GPU 分组配置：4组，每组2卡
GPU_GROUPS = [
    {"id": "group0", "devices": [3, 4], "port": 8081},
    {"id": "group1", "devices": [5, 6], "port": 8082},
]

# 测试参数组合 (VoxCPM 1.5 需要至少 0.5 的显存利用率)
TEST_CONFIGS = [
    TestConfig(0.5, 4, 2048),
    TestConfig(0.5, 8, 2048),
    TestConfig(0.5, 16, 2048),
    TestConfig(0.6, 8, 2048),
    TestConfig(0.6, 16, 2048),
    TestConfig(0.7, 16, 2048),
    TestConfig(0.7, 32, 2048),
    TestConfig(0.8, 32, 2048),
    TestConfig(0.9, 64, 2048),
]

# 参考音频配置
REFERENCE_AUDIO_PATH = "/home/estar/voxcpm-docker/voices/reference/reference_audio.wav"
REFERENCE_TEXT_PATH = "/home/estar/voxcpm-docker/voices/reference/reference_text.txt"
VOICE_NAME = "benchmark_voice"

# 测试文本
TEST_TEXTS = {
    "short": "今天天气真不错，适合出门散步。",
    "medium": "人工智能技术正在快速发展，它已经渗透到我们生活的方方面面。从智能手机的语音助手，到自动驾驶汽车，再到医疗诊断系统，人工智能正在改变世界。",
}

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
        # CUDA_VISIBLE_DEVICES 使用物理GPU ID，VOXCPM_DEVICES 使用相对索引 0,1
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
        print(f"\n启动4组服务 (util={config.gpu_memory_utilization}, seqs={config.max_num_seqs})")
        for group in self.groups:
            group.start(config)
    
    def stop_all(self):
        print("停止所有服务...")
        for group in self.groups:
            group.stop()
        subprocess.run("pkill -f 'uvicorn.*app:app'", shell=True, capture_output=True)
        time.sleep(3)
    
    async def wait_all_ready(self, timeout: int = 300) -> bool:
        print("等待所有服务就绪...")
        results = await asyncio.gather(*[g.wait_ready(timeout) for g in self.groups])
        print(f"  {sum(results)}/4 组服务就绪")
        return all(results)

# ==================== 测试执行 ====================

class VoxCPMBenchmark:
    def __init__(self):
        self.ports = [g["port"] for g in GPU_GROUPS]
        
    def get_gpu_memory(self) -> Dict[str, List[float]]:
        result = {}
        for group in GPU_GROUPS:
            memories = []
            for device_id in group["devices"]:
                proc = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits", f"--id={device_id}"],
                    capture_output=True, text=True)
                memories.append(float(proc.stdout.strip()))
            result[group["id"]] = memories
        return result
    
    async def create_voice_on_all(self) -> bool:
        print(f"\n在4组服务上创建克隆音色...")
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
        print(f"  {sum(results)}/4 组创建成功")
        return all(results)
    
    async def measure_single_rtf(self, port: int, text: str) -> tuple:
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
            return rtf, gen_time, audio_duration
        except Exception as e:
            print(f"  Port {port} RTF测量失败: {e}")
            return -1, -1, -1
    
    async def measure_4group_concurrent(self, text: str) -> tuple:
        async def single_request(port: int):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"http://localhost:{port}/v1/audio/speech",
                        json={"input": text, "model": "voxcpm", "voice": VOICE_NAME},
                        timeout=aiohttp.ClientTimeout(total=300)) as response:
                        await response.read()
                return True
            except:
                return False
        
        start_all = time.perf_counter()
        results = await asyncio.gather(*[single_request(p) for p in self.ports])
        total_time = time.perf_counter() - start_all
        return sum(results), total_time
    
    async def measure_ttfb(self, port: int, text: str) -> float:
        start = time.perf_counter()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"http://localhost:{port}/v1/audio/speech/stream",
                    json={"input": text, "model": "voxcpm", "voice": VOICE_NAME},
                    timeout=aiohttp.ClientTimeout(total=120)) as response:
                    async for chunk in response.content.iter_chunked(1024):
                        return (time.perf_counter() - start) * 1000
        except Exception as e:
            print(f"  TTFB 测量失败: {e}")
        return -1
    
    async def run_test(self, config: TestConfig) -> TestResult:
        result = TestResult(config=config)
        print(f"\n{'='*60}")
        print(f"测试: util={config.gpu_memory_utilization}, seqs={config.max_num_seqs}")
        print('='*60)
        
        try:
            await asyncio.sleep(5)
            
            # 1. 测量显存
            result.gpu_memory_mb = self.get_gpu_memory()
            all_mem = [m for mems in result.gpu_memory_mb.values() for m in mems]
            result.avg_memory_per_gpu = sum(all_mem) / len(all_mem) if all_mem else 0
            print(f"显存: 平均 {result.avg_memory_per_gpu:.0f} MB/卡")
            
            # 2. 创建克隆音色
            if not await self.create_voice_on_all():
                result.success = False
                result.error = "创建克隆音色失败"
                return result
            
            # 3. 测量TTFB
            result.ttfb_ms = await self.measure_ttfb(self.ports[0], TEST_TEXTS["short"])
            print(f"TTFB: {result.ttfb_ms:.0f} ms")
            
            # 4. 测量RTF
            rtf, gen_time, audio_dur = await self.measure_single_rtf(self.ports[0], TEST_TEXTS["medium"])
            result.rtf = rtf
            result.generation_time = gen_time
            result.audio_duration = audio_dur
            print(f"RTF: {result.rtf:.4f} (生成 {gen_time:.2f}s, 音频 {audio_dur:.2f}s)")
            
            # 5. 4组并发测试
            print("4组并发测试...")
            successes, times = [], []
            for i in range(3):
                s, t = await self.measure_4group_concurrent(TEST_TEXTS["medium"])
                successes.append(s)
                times.append(t)
                print(f"  第{i+1}轮: {s}/4成功, 耗时 {t:.2f}s")
            
            result.concurrent_4group_success = int(sum(successes) / len(successes))
            result.concurrent_4group_total_time = sum(times) / len(times)
            result.success = True
            
        except Exception as e:
            result.success = False
            result.error = str(e)
            print(f"测试异常: {e}")
        
        return result

# ==================== 主流程 ====================

def save_results(results, output_dir):
    data = []
    for r in results:
        data.append({
            "config": asdict(r.config),
            "gpu_memory_mb": r.gpu_memory_mb,
            "avg_memory_per_gpu": r.avg_memory_per_gpu,
            "ttfb_ms": r.ttfb_ms,
            "rtf": r.rtf,
            "generation_time": r.generation_time,
            "audio_duration": r.audio_duration,
            "concurrent_4group_success": r.concurrent_4group_success,
            "concurrent_4group_total_time": r.concurrent_4group_total_time,
            "success": r.success,
            "error": r.error,
        })
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def generate_report(results, output_dir):
    lines = ["# VoxCPM 1.5 显存参数测试报告\n",
             f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
             "配置: 4卡分2组, 每组2卡, 使用克隆音色\n",
             "\n## 结果汇总\n",
             "| util | seqs | len | 显存/卡(MB) | TTFB(ms) | RTF | 4组并发 |",
             "|------|------|-----|-------------|----------|-----|---------|"]
    
    for r in results:
        if r.success:
            lines.append(f"| {r.config.gpu_memory_utilization} | {r.config.max_num_seqs} | "
                        f"{r.config.max_model_len} | {r.avg_memory_per_gpu:.0f} | "
                        f"{r.ttfb_ms:.0f} | {r.rtf:.3f} | {r.concurrent_4group_success}/4 ({r.concurrent_4group_total_time:.1f}s) |")
        else:
            lines.append(f"| {r.config.gpu_memory_utilization} | {r.config.max_num_seqs} | "
                        f"{r.config.max_model_len} | FAIL | - | - | {r.error} |")
    
    with open(f"{output_dir}/report.md", "w") as f:
        f.write("\n".join(lines))

async def main():
    output_dir = f"./results/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("VoxCPM 1.5 显存参数优化测试")
    print("配置: 4卡分2组, 每组2卡, 使用克隆音色")
    print("="*60)
    
    service = ServiceManager()
    benchmark = VoxCPMBenchmark()
    results = []
    
    for i, config in enumerate(TEST_CONFIGS):
        print(f"\n\n{'#'*60}")
        print(f"# 配置 {i+1}/{len(TEST_CONFIGS)}: {config}")
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
