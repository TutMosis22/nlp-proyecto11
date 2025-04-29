from src.my_pipeline import image_to_image, show_image, save_image

prompt = "A fantasy castle surrounded by clouds"
init_image_path = "assets/sketch.png"  # Imagen base (debe existir)

result_image = image_to_image(prompt, init_image_path)
show_image(result_image)
save_image(result_image, "metrics/fantasy_castle.png")
