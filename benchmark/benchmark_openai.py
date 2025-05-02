import os
from openai import OpenAI
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

#PROMPT DE PRUEBA
prompt = "A futuristic cityscape at sunset"

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
image.save("dalle_output.png")

print("Imagen generada con OpenAI y guardada como dalle_output.png")
