from diffusers import StableDiffusionPipeline   #MODELO GENERATIVO (PARA GENERAR IMÁGENES)
import torch                                    #PARA MANEJAR TENSORES, MODELOS Y EL HARDWARE
from PIL import Image                           # PARA MOSTRAR IMÁGENES EN LOS NOTEBOOKS
from IPython.display import display
import os
import matplotlib.pyplot as plt

from diffusers import StableDiffusionImg2ImgPipeline
import requests   

from torchvision import transforms, models

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
    
#from diffusers import StableDiffusionImg2ImagePipeline
#import requests    
    
def image_to_image(prompt, init_image_path, strenght = 0.75, guidance_scale = 7.5):
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
        #revision="fp16"
    )
    pipe = pipe.to("cuda")
    
    init_image = Image.open(init_image_path).convert("RGB").resize((512, 512))
    
    image = pipe(prompt=prompt, image=init_image, strength=strenght, guidance_scale=guidance_scale).images[0]
    
    return image

def stylize_image(content_img_path, output_path = "metrics/stylized_image.png"):
    device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #CARAR IMAGEN DE CONTENIDO
    content_image = Image.open(content_img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transform.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_tensor = transform(content_image).unsqueeze(0).to(device)
    
    #CARGAR ESTILO (USARÉ UN MODELO PREENTRENADO ESTILO "CANDY")
    style_model = models.segmentation.deeplabv3_resnet101(pretrained = True).to(device)
    style_model.eval()            #SOLO INFERENCIA, NO ENTRENAMIENTO
    
    #EN ESTE EJEMPLO SE APLICA UN CAMBIO SIMULADO CON UNA CONVOLUCIÓN PARA DEMOSTRAR ESTILO
    # TAMBIÉN LO PODEMOS REEMPALZAR CON MODELOS REALES COMO 'pystiche', 'fast-neural-style'
    with torch.no_grad():
        stylized = torch.nn.functional.avg_pool2d(content_tensor, kernel_size = 3, stride = 1, padding = 1)
        
    #CONVERTIMOS EL TENSOR DE NUEVO A IMAGEN
    stylized_image = stylized.squeeze().permute(1, 2, 0).clamp(0, 255).cpu().numpy().astype("uint8")
    result_image = Image.fromarray(stylized_image)
    result_image.save(output_path)
    
    print(f"Imagen estilizada guardada en {output_path}")
    
    return result_image

#PARA LA MÁSCARA, PERO AL FINAL NO ME CARGÓ :(
from diffusers import StableDiffusionInpaintPipeline

def inpaint_image(prompt, image_path, mask_path, save_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe = pipe.to(device)
    
    init_image = Image.open(image_path).convert("RGB").resize((512, 512))
    mask_image = Image.open(mask_path).convert("RGB").resize((512, 512))
    
    result = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
    
    result.show()
    
    if save_path:
        result.save(save_path)