from src.my_pipeline import image_to_image, show_image, save_image, stylize_image, inpaint_image

prompt = "A man in profile with straight hair and a jacket"
init_image_path = "assets/dibujo_mio.png"  # Imagen base (debe existir)

result_image = image_to_image(prompt, init_image_path)
show_image(result_image)
save_image(result_image, "metrics/man_jacket.png")

original_path = "assets/dibujo_mio.png"
stylized_image = stylize_image(original_path)

prompt = "una version futurista del dibujo original",
image_path="assets/dibujo_mio.png"
mask_path = "assets/mi_dibujo_mascara.png",
save_path = "metrics/dibujo_pintado.png"