import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import os
from PIL import Image

# --- CONFIGURACIÓN ---
CANTIDAD_IMAGENES = 30  # Generaremos 30, tú borrarás las 15 peores
CARPETA_SALIDA = "dataset_sophie_raw"

# ⚠️ TRUCO PRO: Usamos nombres de referencia para fijar la estructura ósea
# Si quieres que Sophie se vea diferente, cambia estos nombres (ej: "Ana de Armas", "Margot Robbie")
ANCHOR_NAMES = "blend of Sydney Sweeney and Ana de Armas"

# Prompt Base (ADN de Sophie)
BASE_PROMPT = (
    f"photo of a woman, {ANCHOR_NAMES}, 20 years old,light brown hair, green eyes, "
    "heart-shaped face, realistic skin texture, natural makeup, closeup, dslr, 8k"
)

# Variaciones para que la IA aprenda la cara en distintos contextos
VARIACIONES = [
    "wearing a white t-shirt, simple background",
    "wearing a black dress, evening light",
    "wearing casual denim jacket, outdoor",
    "wearing red blouse, indoor soft light",
    "wearing gym clothes, fitness center",
    "closeup face portrait, neutral expression, studio lighting",
    "side profile view, looking away, daylight",
    "laughing expression, casual clothes, blurred background",
    "serious expression, professional photo, office background",
    "messy hair bun, pajamas, morning light"
]

# --- INICIO ---
os.makedirs(CARPETA_SALIDA, exist_ok=True)

print("🚀 Iniciando Fábrica de Clones para Dataset...")

# Cargar Modelo (Optimizaciones activadas para tu MSI)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "RunDiffusion/juggernaut-xl-v9",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
pipe.to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

# Generar
contador = 1
for _ in range(3): # Repetimos las variaciones 3 veces
    for estilo in VARIACIONES:
        prompt = f"{BASE_PROMPT}, {estilo}, masterpiece, high quality"
        neg_prompt = "cartoon, anime, 3d, deformed, bad anatomy, makeup heavy, jewelry"
        
        print(f"📸 Generando foto {contador}/{CANTIDAD_IMAGENES}...")
        
        image = pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            width=1024,
            height=1024, # SDXL prefiere 1024x1024 para entrenar
            num_inference_steps=30,
            guidance_scale=7.0
        ).images[0]
        
        nombre_archivo = f"{CARPETA_SALIDA}/sophie_{contador:03d}.png"
        image.save(nombre_archivo)
        print(f"✅ Guardada: {nombre_archivo}")
        contador += 1

print(f"\n🎉 ¡Listo! Revisa la carpeta '{CARPETA_SALIDA}'")