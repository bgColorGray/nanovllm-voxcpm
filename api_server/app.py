"""VoxCPM TTS API Server - OpenAI Compatible
支持 VoxCPM-0.5B (16kHz) 和 VoxCPM 1.5 (44.1kHz)
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from nanovllm_voxcpm.models.voxcpm.server import AsyncVoxCPMServerPool
import base64
from pydantic import BaseModel
from typing import Optional
import torch
import os
import asyncio
import numpy as np
import io
import soundfile
import json

# 智能检测模型路径
def get_default_model_path():
    """自动检测模型路径，优先使用 VoxCPM 1.5"""
    # 1. 环境变量优先
    if os.environ.get("VOXCPM_MODEL_PATH"):
        return os.environ.get("VOXCPM_MODEL_PATH")

    # 2. Docker 环境
    if os.path.exists("/app/VoxCPM1.5/config.json"):
        return "/app/VoxCPM1.5"
    if os.path.exists("/app/VoxCPM-0.5B/config.json"):
        return "/app/VoxCPM-0.5B"

    # 3. 主机环境 - 相对于项目目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)  # nanovllm-voxcpm 的父目录

    # 检查常见位置
    candidates = [
        os.path.join(project_dir, "VoxCPM1.5"),
        os.path.join(project_dir, "VoxCPM-0.5B"),
        os.path.join(os.path.dirname(project_dir), "VoxCPM1.5"),
        os.path.join(os.path.dirname(project_dir), "VoxCPM-0.5B"),
    ]

    for path in candidates:
        if os.path.exists(os.path.join(path, "config.json")):
            return path

    # 默认返回 Docker 路径
    return "/app/VoxCPM1.5"

MODEL_PATH = get_default_model_path()
# 读取显存优化参数（环境变量）
GPU_MEMORY_UTILIZATION = float(os.environ.get("VOXCPM_GPU_MEMORY_UTILIZATION", "0.9"))
MAX_NUM_SEQS = int(os.environ.get("VOXCPM_MAX_NUM_SEQS", "512"))
MAX_MODEL_LEN = int(os.environ.get("VOXCPM_MAX_MODEL_LEN", "4096"))


# 前端路径也做智能检测
def get_frontend_path():
    if os.path.exists("/app/frontend/index.html"):
        return "/app/frontend/index.html"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(os.path.dirname(script_dir), "frontend", "index.html")
    if os.path.exists(local_path):
        return local_path
    return "/app/frontend/index.html"

FRONTEND_PATH = get_frontend_path()

global_instances = {}
voice_cache = {}  # {voice_name: {"prompt_id": str, "prompt_text": str}}

def get_model_info(model_path: str) -> dict:
    """从模型配置文件获取模型信息"""
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)

        # 检测模型版本
        patch_size = config.get("patch_size", 2)
        audio_vae_config = config.get("audio_vae_config", {})
        sample_rate = audio_vae_config.get("sample_rate", 16000) if audio_vae_config else 16000

        # 根据 patch_size 判断版本
        if patch_size == 4:
            model_name = "voxcpm-1.5"
        else:
            model_name = "voxcpm-0.5b"

        return {
            "name": model_name,
            "sample_rate": sample_rate,
            "patch_size": patch_size,
        }

    # 默认值 (VoxCPM-0.5B)
    return {
        "name": "voxcpm-0.5b",
        "sample_rate": 16000,
        "patch_size": 2,
    }

def get_gpu_devices():
    env_devices = os.environ.get("VOXCPM_DEVICES")
    if env_devices:
        return [int(d) for d in env_devices.split(",")]
    gpu_count = torch.cuda.device_count()
    return list(range(gpu_count)) if gpu_count > 0 else [0]

async def warmup_all_gpus(gpu_count: int):
    """并发预热所有 GPU"""
    async def single_warmup(i):
        async for _ in global_instances["server"].generate(
            target_text=f"预热{i}",
            prompt_latents=None,
            prompt_text="",
            prompt_id=None,
            max_generate_length=50,
            temperature=1.0,
            cfg_value=1.5,
        ):
            pass
    await asyncio.gather(*[single_warmup(i) for i in range(gpu_count)])
    print(f"[VoxCPM] 所有 {gpu_count} 张 GPU 预热完成")

@asynccontextmanager
async def lifespan(app: FastAPI):
    devices = get_gpu_devices()
    print(f"[VoxCPM] 使用 GPU 设备: {devices}")
    print(f"[VoxCPM] 模型路径: {MODEL_PATH}")

    # 获取模型信息
    model_info = get_model_info(MODEL_PATH)
    global_instances["model_info"] = model_info
    print(f"[VoxCPM] 模型版本: {model_info['name']}, 采样率: {model_info['sample_rate']}Hz")

    server = AsyncVoxCPMServerPool(
        model_path=MODEL_PATH,
        devices=devices,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_num_seqs=MAX_NUM_SEQS,
        max_model_len=MAX_MODEL_LEN,
    )
    global_instances["server"] = server

    # 智能预热
    await warmup_all_gpus(len(devices))

    yield
    global_instances.clear()
    voice_cache.clear()

app = FastAPI(title="VoxCPM TTS API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Sample-Rate", "X-Channels", "X-Bit-Depth"],
)

def get_sample_rate() -> int:
    """获取当前模型的采样率"""
    return global_instances.get("model_info", {}).get("sample_rate", 16000)

# ==================== Web UI ====================

@app.get("/", response_class=HTMLResponse)
async def web_ui():
    if os.path.exists(FRONTEND_PATH):
        with open(FRONTEND_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h1>VoxCPM TTS API</h1><p>Frontend not found</p>")

# ==================== Health & Info ====================

@app.get("/health")
async def health():
    model_info = global_instances.get("model_info", {"name": "unknown", "sample_rate": 16000})
    return {
        "status": "ok",
        "model": model_info["name"],
        "sample_rate": model_info["sample_rate"],
        "voices_count": len(voice_cache)
    }

@app.get("/voices")
async def list_voices():
    return {"voices": list(voice_cache.keys()), "count": len(voice_cache)}

# ==================== Voice Management ====================

class CreateVoiceRequest(BaseModel):
    voice_name: str
    prompt_wav_base64: Optional[str] = None
    prompt_wav_path: Optional[str] = None
    prompt_wav_format: str = "wav"
    prompt_text: str
    replace: bool = False

@app.post("/v1/voices")
async def create_voice(request: CreateVoiceRequest):
    if request.voice_name in voice_cache and not request.replace:
        raise HTTPException(status_code=409, detail=f"Voice '{request.voice_name}' already exists")

    wav_data = None
    wav_format = request.prompt_wav_format

    if request.prompt_wav_base64:
        try:
            wav_data = base64.b64decode(request.prompt_wav_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 data: {e}")
    elif request.prompt_wav_path:
        if not os.path.exists(request.prompt_wav_path):
            raise HTTPException(status_code=404, detail=f"Audio file not found: {request.prompt_wav_path}")
        try:
            with open(request.prompt_wav_path, "rb") as f:
                wav_data = f.read()
            ext = os.path.splitext(request.prompt_wav_path)[1].lower().lstrip(".")
            if ext in ["wav", "mp3", "flac", "ogg"]:
                wav_format = ext
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read audio file: {e}")
    else:
        raise HTTPException(status_code=400, detail="Must provide either prompt_wav_base64 or prompt_wav_path")

    prompt_text = request.prompt_text
    if os.path.exists(prompt_text):
        try:
            with open(prompt_text, "r", encoding="utf-8") as f:
                prompt_text = f.read().strip()
        except:
            pass

    try:
        server = global_instances["server"]
        prompt_id = await server.add_prompt(wav_data, wav_format, prompt_text)

        voice_cache[request.voice_name] = {
            "prompt_id": prompt_id,
            "prompt_text": prompt_text
        }

        return {"status": "success", "voice_name": request.voice_name, "prompt_id": prompt_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/v1/voices/{voice_name}")
async def delete_voice(voice_name: str):
    if voice_name not in voice_cache:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found")

    try:
        prompt_id = voice_cache[voice_name]["prompt_id"]
        server = global_instances["server"]
        await server.remove_prompt(prompt_id)
        del voice_cache[voice_name]
        return {"status": "success", "message": f"Voice '{voice_name}' deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== OpenAI Compatible TTS API ====================

class SpeechRequest(BaseModel):
    model: str = "voxcpm"
    input: str
    voice: Optional[str] = None
    response_format: str = "wav"
    max_length: int = 2000
    temperature: float = 1.0
    cfg_value: float = 1.5

@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    server = global_instances["server"]
    sample_rate = get_sample_rate()

    prompt_id = None
    prompt_text = ""
    if request.voice and request.voice in voice_cache:
        prompt_id = voice_cache[request.voice]["prompt_id"]

    chunks = []
    async for chunk in server.generate(
        target_text=request.input,
        prompt_latents=None,
        prompt_text=prompt_text,
        prompt_id=prompt_id,
        max_generate_length=request.max_length,
        temperature=request.temperature,
        cfg_value=request.cfg_value,
    ):
        chunks.append(chunk)

    audio = np.concatenate(chunks)

    buffer = io.BytesIO()
    soundfile.write(buffer, audio, sample_rate, format=request.response_format.upper())
    buffer.seek(0)

    media_types = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac"}
    return Response(
        content=buffer.read(),
        media_type=media_types.get(request.response_format, "audio/wav"),
        headers={"Content-Disposition": f"attachment; filename=speech.{request.response_format}"}
    )

@app.post("/v1/audio/speech/stream")
async def create_speech_stream(request: SpeechRequest):
    server = global_instances["server"]
    sample_rate = get_sample_rate()

    prompt_id = None
    prompt_text = ""
    if request.voice and request.voice in voice_cache:
        prompt_id = voice_cache[request.voice]["prompt_id"]

    async def audio_generator():
        chunk_index = 0
        async for chunk in server.generate(
            target_text=request.input,
            prompt_latents=None,
            prompt_text=prompt_text,
            prompt_id=prompt_id,
            max_generate_length=request.max_length,
            temperature=request.temperature,
            cfg_value=request.cfg_value,
        ):
            # 确保是 float32 类型
            chunk = chunk.astype(np.float32)

            # 安全转换：clipping 防止溢出
            pcm_16bit = np.clip(chunk * 32767, -32768, 32767).astype(np.int16)
            yield pcm_16bit.tobytes()
            await asyncio.sleep(0)  # 强制让出控制权，确保数据及时发送
            chunk_index += 1

    return StreamingResponse(
        audio_generator(),
        media_type="application/octet-stream",
        headers={"X-Sample-Rate": str(sample_rate), "X-Channels": "1", "X-Bit-Depth": "16"}
    )

# ==================== Legacy API (Backward Compatibility) ====================

class GenerateRequest(BaseModel):
    target_text: str
    prompt_id: Optional[str] = None
    prompt_text: str = ""
    max_generate_length: int = 2000
    temperature: float = 1.0
    cfg_value: float = 1.5

@app.post("/generate")
async def generate_legacy(request: GenerateRequest):
    server = global_instances["server"]
    sample_rate = get_sample_rate()

    chunks = []
    async for chunk in server.generate(
        target_text=request.target_text,
        prompt_latents=None,
        prompt_text=request.prompt_text,
        prompt_id=request.prompt_id,
        max_generate_length=request.max_generate_length,
        temperature=request.temperature,
        cfg_value=request.cfg_value,
    ):
        chunks.append(chunk)

    audio = np.concatenate(chunks)
    buffer = io.BytesIO()
    soundfile.write(buffer, audio, sample_rate, format="WAV")
    buffer.seek(0)

    return Response(content=buffer.read(), media_type="audio/wav")
