import time
import os
import csv
import psutil
import torch
from diffusers import StableDiffusionPipeline

# Lista de prompts definidos directamente en el script
prompts = [
    "A futuristic cityscape at sunset",
    "A fantasy castle in the mountains, digital art",
    "A cat playing chess with a dog, photorealistic"
]

# SE CARGA EL MODELO DESDE HUGGING FACE (demora un poco)
print("Cargando modelo desde transformers...")
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2", torch_dtype=torch.float16)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
print("Modelo cargado.")

#RUTA PARA GUARDAR IMÁGENES
output_dir = "benchmark"
os.makedirs(output_dir, exist_ok=True)

#RUTA DEL CSV PARA GUARDAR MÉTRICAS
csv_file = os.path.join("metrics", "resultados.csv")
os.makedirs("metrics", exist_ok=True)

# Obtener uso de memoria antes de la generación
mem_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # en MB
start_time = time.time()

# SE GENERAN LAS IMÁGENES Y SE GUARDAN MÉTRICAS POR PROMPT
for i, prompt in enumerate(prompts):
    prompt_start = time.time()
    image = pipe(prompt).images[0]
    prompt_time = time.time() - prompt_start

    image_filename = os.path.join(output_dir, f"transformers_image_{i+1}.png")
    image.save(image_filename)

    #MANDAMOS LOS RESULTADOS A NUESTRO ARCHIVO .CSV
    with open(csv_file, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["transformers", prompt, f"{prompt_time:.2f}", "NA", image_filename])

#NUEVAMENTE MEDIMOS LA MEMOIA Y EL TIEMPO TOTAL
end_time = time.time()
mem_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # en MB
total_time = end_time - start_time

print(f"\nTiempo total: {total_time:.2f} segundos")
print(f"Memoria antes: {mem_before:.2f} MB")
print(f"Memoria después: {mem_after:.2f} MB")
