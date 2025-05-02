import openai
import os
from dotenv import load_dotenv

#CARGAMOS LAS VARIABLES DE ENTORNO
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_text_with_openai(prompt, model="gpt-3.5-turbo", max_tokens=100):
    """
    Genera texto usando la API de OpenAI con ChatCompletion.
    """
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7
    )
    
    return response.choices[0].message.content.strip()
