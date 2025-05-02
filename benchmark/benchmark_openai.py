import os
import time
import csv
from openai import OpenAI
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import unicodedata
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("OPENAI_API_KEY")
api_key_clean = unicodedata.normalize("NFKD", api_key).encode("ascii", "ignore").decode("ascii")

client = OpenAI(api_key=api_key_clean)

#PROMPT DE PRUEBA
prompt = "A futuristic cityscape at sunset"

start_time = time.time()

# LLAMAMOS A LA API DE OPEN AI PARA GEENRAR UNA IMAGEN
response = client.images.generate(
    model="dall-e-3",
    prompt=prompt,
    size="1024x1024",
    quality="standard",
    n=1,
)

#OBTENEMOS UNA URL DE LA IMAGEN QUE SE GENERA
image_url = response.data[0].url

#LA DESCARGAMOS Y LUEGO LA GUARDAMOS
image_response = requests.get(image_url)
image = Image.open(BytesIO(image_response.content))
output_path = "dalle_output2.png"
image.save(output_path)

#MÉTRICAS
end_time = time.time()
elapsed_time = end_time - start_time
file_size_kb = os.path.getsize(output_path) / 1024

print("Imagen generada con OpenAI y guardada como dalle_output2.png")
print(f"Tiempo de inferencia (segundos): {elapsed_time:.2f}")
print(f"Tamaño del archivo (KB): {file_size_kb:.2f}")

csv_file = "../metrics/resultados.csv"
os.makedirs("metrics", exist_ok = True)
header = ["\nmodelo", "prompt", "tiempo_segundos", "tamano_kb"]

# Verificar si el archivo existe
file_exists = os.path.isfile(csv_file)

with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(header)
    writer.writerow(["openai-dalle3", prompt, f"{elapsed_time:.2f}", f"{file_size_kb:.2f}"])