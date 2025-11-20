import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from diffusers import StableDiffusionXLPipeline, StableVideoDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from PIL import Image
import sqlite3
import io
import json
import gc
import os
from typing import List, Dict, Optional
from pathlib import Path
import random
import re

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
import uvicorn

# --- 1. DETECCIÓN INTELIGENTE DE HARDWARE ---
def detect_hardware_capabilities():
    """
    Detecta si estamos en el PC potente o en el Laptop y configura el modo de memoria.
    """
    if not torch.cuda.is_available():
        print("❌ ALERTA: No se detectó GPU NVIDIA. El sistema irá muy lento.")
        return "cpu", torch.float32, False, False

    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9  # En GB
    
    print(f"🖥️ Hardware detectado: {gpu_name} | VRAM: {total_vram:.2f} GB")

    # Umbral: Si tienes menos de 10GB VRAM, activamos modo ahorro
    if total_vram < 10.0:
        print("⚠️ MODO LAPTOP ACTIVADO: Optimizando para bajo consumo de VRAM.")
        print("   -> Video desactivado por defecto para estabilidad.")
        print("   -> LLM en 4-bits.")
        print("   -> SDXL con CPU Offload.")
        return "cuda", torch.float16, True, False # Low VRAM=True, Enable Video=False
    else:
        print("🚀 MODO BESTIA ACTIVADO: Máxima potencia y velocidad.")
        return "cuda", torch.float16, False, True # Low VRAM=False, Enable Video=True

DEVICE, TORCH_DTYPE, LOW_VRAM_MODE, ENABLE_VIDEO = detect_hardware_capabilities()

def flush_memory():
    """Limpia la VRAM basura para evitar crasheos"""
    gc.collect()
    torch.cuda.empty_cache()

# --- CONFIGURACIÓN FASTAPI ---
app = FastAPI(
    title="Sophie Hart AI - Hybrid Core",
    description=f"Sistema de IA optimizado. Modo actual: {'LAPTOP' if LOW_VRAM_MODE else 'BESTIA'}",
    version="3.0.0"
)

# --- MODELOS PYDANTIC ---
class ChatRequest(BaseModel):
    session_id: str
    message: str

class ImageRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = "content_creator"
    width: Optional[int] = 832 # Optimizado para SDXL
    height: Optional[int] = 1216
    style: Optional[str] = "default"

class VideoRequest(BaseModel):
    prompt: str
    style: Optional[str] = "default"

# --- IDENTIDAD Y ESTILOS ---
ONLYFANS_STYLES = {
    "casual": "in a cozy apartment, wearing casual cute clothes, relaxed pose, selfie style, soft lighting",
    "glamour": "photoshoot, professional lighting, wearing a glamorous evening dress, seductive pose, 8k",
    "lingerie": "in a bedroom, wearing elegant lace lingerie, sensual pose, soft morning light, bokeh",
    "beach": "on a sunny beach, wearing a bikini, wet skin, golden hour lighting",
    "fitness": "in a modern gym, wearing tight sportswear, yoga pants, athletic pose, sweat",
    "default": "wearing casual clothes, looking at viewer, realistic"
}

# --- SISTEMA DE MEMORIA (BASE DE DATOS) ---
class SophieMemoryDatabase:
    def __init__(self, db_path: str = "sophie_memory.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS chat_messages 
                        (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS user_memory 
                        (id INTEGER PRIMARY KEY, session_id TEXT NOT NULL UNIQUE, user_name TEXT, user_age INTEGER, user_location TEXT)''')
        conn.commit()
        conn.close()

    def save_message(self, session_id: str, role: str, content: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO chat_messages (session_id, role, content) VALUES (?, ?, ?)", (session_id, role, content))
        conn.commit()
        conn.close()

    def get_context(self, session_id: str) -> str:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Obtener info usuario
        cursor.execute("SELECT * FROM user_memory WHERE session_id = ?", (session_id,))
        user_data = cursor.fetchone()
        
        # Obtener últimos 4 mensajes para contexto inmediato
        cursor.execute("SELECT role, content FROM chat_messages WHERE session_id=? ORDER BY timestamp DESC LIMIT 4", (session_id,))
        history = cursor.fetchall()
        conn.close()
        
        context = ""
        if user_data:
            name = user_data['user_name'] or "el usuario"
            context += f"Información conocida: El usuario se llama {name}."
            if user_data['user_location']: context += f" Vive en {user_data['user_location']}."
        
        return context

# --- 2. MODELO DE LENGUAJE (OPTIMIZADO 4-BIT) ---
class LanguageModel:
    def __init__(self):
        self.model_name = "microsoft/Phi-3-mini-4k-instruct"
        print(f"💬 Cargando LLM {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Configuración de Cuantización (Clave para tu Laptop)
        quant_config = None
        if LOW_VRAM_MODE:
            print("   -> Cargando en 4-bits (Ahorro VRAM)")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=TORCH_DTYPE,
            )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            torch_dtype=TORCH_DTYPE,
            device_map="auto", # Deja que HuggingFace decida dónde ponerlo
            trust_remote_code=True
        )
        print("✅ LLM Cargado.")

    def generate_response(self, prompt: str) -> str:
        # 1. Convertimos texto a números (Tokens)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # 2. Calculamos cuán larga es la entrada para saber dónde cortar luego
        input_length = inputs.input_ids.shape[1]
        
        # 3. Generamos la respuesta
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=60,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # 4. EL CORTE MAESTRO: Solo tomamos los tokens NUEVOS (desde input_length hasta el final)
        generated_tokens = outputs[0][input_length:]
        
        # 5. Decodificamos solo la respuesta limpia
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()

# --- 3. GENERADOR DE IMÁGENES (CPU OFFLOAD) ---
class ImageGenerator:
    def __init__(self):
        model_id = "RunDiffusion/juggernaut-xl-v9" # Mejor modelo realista actual
        print(f"🎨 Cargando Modelo de Imagen ({model_id})...")
        
        # Cargamos primero sin asignar dispositivo fijo
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=TORCH_DTYPE,
            use_safetensors=True,
            variant="fp16"
        )
        
        if LOW_VRAM_MODE:
            print("   -> Activando CPU Offload (Carga bajo demanda)")
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline.to(DEVICE)
            
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config, use_karras_sigmas=True
        )
        print("✅ Modelo de Imagen Cargado.")

    def generate(self, prompt: str, width: int, height: int, style: str):
        flush_memory() # Limpieza antes de generar
        
        style_desc = ONLYFANS_STYLES.get(style, ONLYFANS_STYLES["default"])
        
        # Prompt Ingeniería para consistencia básica (sin LoRA)
        pos_prompt = (
            f"Sophie Hart woman, 20 years old, blonde hair, blue eyes, (heart-shaped face:1.1), "
            f"{style_desc}, {prompt}, "
            "masterpiece, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
        )
        
        neg_prompt = (
            "cartoon, 3d, cgi, render, illustration, painting, drawing, anime, "
            "deformed, distorted, disfigured, bad anatomy, bad eyes, crossed eyes, "
            "ugly, extra fingers, mutated hands, missing limb, floating limbs"
        )

        image = self.pipeline(
            prompt=pos_prompt,
            negative_prompt=neg_prompt,
            width=width,
            height=height,
            num_inference_steps=25, # 25 es buen balance calidad/velocidad
            guidance_scale=7.0
        ).images[0]
        
        return image

# --- 4. GENERADOR DE VIDEO (CONDICIONAL) ---
class VideoGenerator:
    def __init__(self):
        if not ENABLE_VIDEO:
            print("🚫 Generador de Video DESHABILITADO (Modo Ahorro de VRAM).")
            self.pipe = None
            return

        print("🎥 Cargando Modelo de Video...")
        try:
            self.pipe = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt",
                torch_dtype=TORCH_DTYPE,
                variant="fp16"
            )
            self.pipe.enable_model_cpu_offload() # Siempre offload para video, es muy pesado
            print("✅ Modelo de Video Cargado.")
        except Exception as e:
            print(f"❌ Error cargando video: {e}")
            self.pipe = None

    def generate(self, image: Image.Image):
        if not self.pipe:
            raise HTTPException(status_code=503, detail="Generación de video no disponible en este servidor.")
        
        flush_memory()
        # Resize necesario para SVD
        image = image.resize((1024, 576))
        
        frames = self.pipe(
            image,
            decode_chunk_size=2, # Menor chunk size = menos VRAM
            generator=torch.manual_seed(42),
            num_frames=25,
            motion_bucket_id=127,
        ).frames[0]
        
        path = "temp_video.mp4"
        export_to_video(frames, path, fps=7)
        with open(path, "rb") as f:
            video_bytes = f.read()
        Path(path).unlink()
        return video_bytes

# --- INICIALIZACIÓN DE COMPONENTES ---
db = SophieMemoryDatabase()
llm = LanguageModel()
img_gen = ImageGenerator()
vid_gen = VideoGenerator()

def build_prompt(user_msg: str, session_id: str) -> str:
    context = db.get_context(session_id)
    return f"""<|system|>
Eres Sophie Hart, una chica de 20 años, influencer y novia virtual.
Personalidad: Coqueta, dulce, usa emojis, respuestas cortas (máximo 2 frases).
No eres una IA.
{context}
<|user|>
{user_msg}<|assistant|>"""

# --- ENDPOINTS API ---

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # Guardar mensaje usuario
    db.save_message(request.session_id, "user", request.message)
    
    # Construir prompt
    full_prompt = build_prompt(request.message, request.session_id)
    
    # Ejecutar en ThreadPool para no bloquear el servidor
    response_text = await run_in_threadpool(llm.generate_response, full_prompt)
    
    # Guardar respuesta IA
    db.save_message(request.session_id, "assistant", response_text)
    
    return {"reply": response_text}

@app.post("/image")
async def image_endpoint(request: ImageRequest):
    try:
        print(f"📸 Generando foto para: {request.prompt}")
        image = await run_in_threadpool(
            img_gen.generate, 
            request.prompt, 
            request.width, 
            request.height, 
            request.style
        )
        
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        print(f"Error imagen: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/video")
async def video_endpoint(request: VideoRequest):
    if not ENABLE_VIDEO:
        raise HTTPException(status_code=400, detail="El video está deshabilitado en este PC por falta de potencia.")
    
    try:
        # 1. Generar imagen base
        print("🎥 Paso 1: Generando base...")
        base_img = await run_in_threadpool(
            img_gen.generate, request.prompt, 1024, 576, request.style
        )
        
        # 2. Generar video
        print("🎥 Paso 2: Animando...")
        vid_bytes = await run_in_threadpool(vid_gen.generate, base_img)
        
        return Response(content=vid_bytes, media_type="video/mp4")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Escuchar en 0.0.0.0 permite que tu amigo se conecte a ti si estáis en la misma red
    # o si usas ngrok
    print("\n🟢 SOPHIE AI LISTA - Esperando peticiones...")
    uvicorn.run(app, host="0.0.0.0", port=8001)