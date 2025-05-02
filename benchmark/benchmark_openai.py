import os
import openai
import requests
from PIL import Image
from io import BytesIO

openai.api_key = os.getenv("OPENAI_API_KEY")

#PROMPT DE PRUEBA
prompt = "A futuristic cityscape at sunset"

# LLAMAMOS A LA API DE OPEN AI PARA GEENRAR UNA IMAGEN
response = openai.images.generate(
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

print("Imagen generada con OpenAI y guardada como benchmark/dalle_output.png")
