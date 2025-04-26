from diffusers import StableDiffusionPipeline   #MODELO GENERATIVO (PARA GENERAR IMÁGENES)
import torch                                    #PARA MANEJAR TENSORES, MODELOS Y EL HARDWARE
from PIL import Image                           # PARA MOSTRAR IMÁGENES EN LOS NOTEBOOKS
from IPython.display import display
import os
import matplotlib.pyplot as plt

def load_model(model_name = "runwayml/stable-diffusion-v1-5", use_cuda=True):
    """
    Carga el modelo de difusión estable desde Hugging Face y lo envía a la GPU si está disponible
    
    Parámetros:
    - model_name (str): Nombre del modelo en Hugging Face
    - use_cuda (bool): Si True y hay GPU disponible, mueve el modelo a CUDA
    
    Retorna:
    - pipe: pipeline ya cargado y listo para usar
    """
    
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    
    if use_cuda and torch.cuda.is_available():
        pipe = pipe.to("cuda")
    else:
        pipe =pipe.to("cpu")
    
    return pipe

def generate_image(pipe, prompt, num_inference_steps=30, height = 512, width = 512):
    """
    Usa el modelo para generar una imagen a partir de un texto
    
    Parámetros:
    - pipe: El pipeline de difusión cargado
    - prompt (str): El texto que describe la imagen.
    -num_inference_steps (int): cantidad de pasos de inferencia (mayor = mejor calidad)
    - height (init): altura de la imagen generada (pixeles)
    - width (int): ancho de la imagen generada (pixeles)
    
    Retorna:
    - image(PIL.Image): La imagen ya generada
    """
    output = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        height = height,
        width = width
    )
    image = output.images[0]
    
    return image

def show_image(image):
    """
    Se muestra la imagen generada usando matplotlib.
    
    Parámetros:
    - image (PIL.Image): Imagen a mostrar
    """
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
#    try:
#        display(image)
#    except:
#        image.show()

def save_image(image, output_path):
    """ 
    Guarda una imagen generada en un archivo
    
    Parámetros:
    - image: imagen generada (PIL.Image)
    - output_path (str): ruta completa del archivo a guardar (debe terminar en .png o .jpg)
    """
    #CREA EL DIRECTORIO SI NO EXISTE
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    image.save(output_path)