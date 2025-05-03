import time
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import psutil           #LO USAREMOS PARA MEDIR EL USO DE MEMORIA
import os

#SE CARGA LA IMAGEN DE EJEMPLO
init_image = Image.open("../metrics/cityscape.png").convert("RGB").resize((512, 512))

#DEFINO EL PROMPT
prompt = "A futuristic cityscape at sunset"

#SE MIDE EL USO DE MEMORIA ANTES
process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / 1024 / 1024  # en MB

#MEDIMOS EL TIEMPO DE CARGA E INFERENCIA
start_time = time.time()

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32
).to("cuda" if torch.cuda.is_available() else "cpu")

output = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images[0]

end_time = time.time()          #PARA MEDIR EL TIEMPO DE INFERENCIA
mem_after = process.memory_info().rss / 1024 / 1024

#SE GUARDA LA IMGEN RESULTANTE
output.save("diffusers_output.png")

print("=== BENCHMARK DIFFUSERS ===")
print(f"Tiempo total (s): {end_time - start_time:.2f}")
print(f"Uso de memoria antes (MB): {mem_before:.2f}")
print(f"Uso de memoria después (MB): {mem_after:.2f}")
