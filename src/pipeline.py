from diffusers import StableDiffusionPipeline   #MODELO GENERATIVO (PARA GENERAR IMÁGENES)
import torch                                    #PARA MANEJAR TENSORES, MODELOS Y EL HARDWARE
from PIL import Image                           # PARA MOSTRAR IMÁGENES EN LOS NOTEBOOKS
from IPython.display import display

def load_model(model_name = "runwayml/stable-diffusion-v1-5"):
    """
    Carga el modelo de difusión estable desde Hugging Face y lo envía a la GPU si está disponible
    
    Parámetros:
    - model_name (str): Nombre del modelo en Hugging Face
    
    Retorna:
    - pipe: pipeline ya cargado y listo para usar
    """
    
    pipe = StableDiffusionPipeline.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    
    return pipe

def generate_image(pipe, prompt):
    """
    Usa el modelo para generar una imagen a partir de un texto
    
    Parámetros:
    - pipe: El pipeline de difusión cargado
    - prompt (str): El texto que describe la imagen
    
    Retorna:
    - image(PIL.Image): La imagen ya generada
    """
    image = pipe(prompt).images[0]
    
    return image

def show_image(image):
    """
    Se muestra la imagen generada.
    
    Parámetros:
    - image (PIL.Image): Imagen a mostrar
    """
    
    try:
        display(image)
    except:
        image.show()
    