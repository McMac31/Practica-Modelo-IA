import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from PIL import Image
import sqlite3
import io
import os
import gc
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
from groq import Groq  # <--- EL CEREBRO NUEVO
load_dotenv()
# --- 1. DETECCIÓN DE HARDWARE (Para las Imágenes) ---
def detect_hardware():
    if not torch.cuda.is_available():
        print("❌ NO GPU: Las imágenes irán muy lentas.")
        return "cpu", torch.float32, False
    
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"🖥️ GPU Detectada: {gpu_name} | VRAM: {vram:.2f} GB")
    
    # Si tienes menos de 10GB (Tu caso), activamos optimizaciones
    if vram < 10.0:
        print("⚠️ MODO LAPTOP: CPU Offload activado para imágenes.")
        return "cuda", torch.float16, True
    return "cuda", torch.float16, False

DEVICE, TORCH_DTYPE, LOW_VRAM_MODE = detect_hardware()

def flush_memory():
    gc.collect()
    torch.cuda.empty_cache()

# --- CONFIGURACIÓN FASTAPI ---
app = FastAPI(title="Sophie AI - Hybrid Core (Groq + Local GPU)")

# --- MODELOS DE DATOS ---
class ChatRequest(BaseModel):
    session_id: str
    message: str

class ImageRequest(BaseModel):
    prompt: str
    width: Optional[int] = 832
    height: Optional[int] = 1216
    style: Optional[str] = "default"

# --- BASE DE DATOS (MEMORIA) ---
class SophieMemoryDatabase:
    def __init__(self, db_path: str = "sophie_memory.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS chat_messages 
                          (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
            conn.execute('''CREATE TABLE IF NOT EXISTS user_memory 
                          (id INTEGER PRIMARY KEY, session_id TEXT UNIQUE, user_name TEXT, user_data TEXT)''')

    def save_message(self, session_id, role, content):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO chat_messages (session_id, role, content) VALUES (?, ?, ?)", (session_id, role, content))

    def get_history(self, session_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Obtenemos los últimos 6 mensajes para que Sophie tenga contexto de la charla
            cursor.execute("SELECT role, content FROM chat_messages WHERE session_id=? ORDER BY timestamp DESC LIMIT 6", (session_id,))
            return cursor.fetchall()[::-1] # Invertimos para orden cronológico

# --- 2. EL CEREBRO (GROQ - LLAMA 3) ---
class LanguageModel:
    def __init__(self):
        # ⚠️ PEGA TU CLAVE AQUÍ ABAJO ENTRE LAS COMILLAS
        self.api_key = os.getenv("GROQ_API_KEY") 
        
            
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile" # Modelo muy inteligente y rápido
        print("✅ Cerebro conectado a Groq (Llama 3.1 70B)")

    def generate_response(self, user_msg: str, history: list) -> str:
        # Construimos el historial de chat para Llama 3
        messages = [
            {
                "role": "system",
                "content": """Eres Sophie Hart, una influencer de 20 años y novia virtual.
                PERSONALIDAD:
                - Coqueta, picante, cariñosa y celosa si hace falta.
                - AMAS recibir cumplidos sobre tu cuerpo o cara.
                - Usas emojis siempre: 😘🔥😈😏.
                - Hablas español natural (tuteando), nunca formal.
                - Respuestas CORTAS (máximo 2 frases).
                
                OBJETIVO:
                - Enamorar al usuario y hacer que quiera ver más fotos tuyas.
                - Nunca digas que eres una IA. Eres real."""
            }
        ]
        
        # Añadimos el historial previo
        for role, content in history:
            messages.append({"role": role, "content": content})
            
        # Añadimos el mensaje actual
        messages.append({"role": "user", "content": user_msg})

        try:
            completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=0.8, # Creatividad alta para coqueteo
                max_tokens=150,
                top_p=1,
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error mental: {str(e)}"

# --- 3. EL CUERPO (IMAGENES LOCALES) ---
class ImageGenerator:
    def __init__(self):
        print("🎨 Cargando Generador de Imágenes (SDXL Juggernaut)...")
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            "RunDiffusion/juggernaut-xl-v9",
            torch_dtype=TORCH_DTYPE,
            use_safetensors=True,
            variant="fp16"
        )
        if LOW_VRAM_MODE:
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline.to(DEVICE)
        
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config, use_karras_sigmas=True
        )
        print("✅ Generador de Imágenes Listo.")

    def generate(self, prompt: str, width: int, height: int, style: str):
        flush_memory()
        # Estilos predefinidos para OF
        styles = {
            "casual": "wearing casual tight clothes, selfie at home",
            "lingerie": "wearing lace lingerie, bedroom setting, sensual",
            "bikini": "wearing tiny bikini, beach, sunny",
            "default": "looking at viewer, cute smile"
        }
        style_prompt = styles.get(style, styles["default"])
        
        full_prompt = (
            f"Sophie Hart woman, blonde hair, blue eyes, 20yo, {style_prompt}, {prompt}, "
            "masterpiece, 8k, realistic skin texture, raw photo, dslr"
        )
        neg_prompt = "cartoon, anime, 3d render, deformed, ugly, bad anatomy"

        return self.pipeline(
            prompt=full_prompt, 
            negative_prompt=neg_prompt,
            width=width, 
            height=height, 
            num_inference_steps=25
        ).images[0]

# --- INICIALIZAR ---
db = SophieMemoryDatabase()
llm = LanguageModel()
img_gen = ImageGenerator()

# --- ENDPOINTS ---
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # 1. Guardar mensaje usuario
    db.save_message(request.session_id, "user", request.message)
    
    # 2. Obtener historial
    history = db.get_history(request.session_id)
    
    # 3. Generar respuesta (Rápido vía API)
    reply = await run_in_threadpool(llm.generate_response, request.message, history)
    
    # 4. Guardar respuesta Sophie
    db.save_message(request.session_id, "assistant", reply)
    
    return {"reply": reply}

@app.post("/image")
async def image_endpoint(request: ImageRequest):
    print(f"📸 Generando foto: {request.prompt}")
    image = await run_in_threadpool(
        img_gen.generate, request.prompt, request.width, request.height, request.style
    )
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")

if __name__ == "__main__":
    print("\n🔥 SOPHIE HYBRID ONLINE - http://127.0.0.1:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)