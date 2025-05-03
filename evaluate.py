import os
import pandas as pd
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt

#UBICACIÓN DE LA IMAGEN BASE Y CARPETA DE LAS IMÁGENES GENERADAS
BASE_IMAGE_PATH = "metrics/cityscape.png"
IMAGES_FOLDER = "benchmark"
OUTPUT_CSV = "metrics/evaluacion.csv"
IMAGE_SIZE = (512, 512)

#LEE LA IMAGEN BASE
if not os.path.exists(BASE_IMAGE_PATH):
    raise FileNotFoundError(f"No se encontró la imagen base en: {BASE_IMAGE_PATH}")
base_img = imread(BASE_IMAGE_PATH)
base_img = resize(base_img, IMAGE_SIZE, anti_aliasing=True)

#LEER IMÁGENES GENERADAS POR MODELOS
files = [f for f in os.listdir(IMAGES_FOLDER) if f.endswith((".png", ".jpg"))]

results = []

for file in files:
    model_name = file.split("_")[0]
    test_img_path = os.path.join(IMAGES_FOLDER, file)
    test_img = imread(test_img_path)
    test_img = resize(test_img, IMAGE_SIZE, anti_aliasing=True)

    # CALCULAR LAS MÉTRICAS
    ssim_val = ssim(base_img, test_img, channel_axis=-1, data_range=1.0)
    psnr_val = psnr(base_img, test_img, data_range=1.0)

    results.append({
        "imagen_generada": file,
        "modelo": model_name,
        "ssim": ssim_val,
        "psnr": psnr_val
    })

#GUARDAR RESULTADOS
os.makedirs("metrics", exist_ok=True)
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Evaluación completada. Resultados guardados en {OUTPUT_CSV}")

#SE GRAFICAN LOS RESULTADOS
plt.figure(figsize=(10, 5))
modelos = df["modelo"].unique()

#PROMEDIO POR MODELO
mean_df = df.groupby("modelo")[["ssim", "psnr"]].mean().reset_index()

x = np.arange(len(modelos))
width = 0.35

plt.bar(x - width/2, mean_df["ssim"], width, label='SSIM', color='skyblue')
plt.bar(x + width/2, mean_df["psnr"], width, label='PSNR', color='lightgreen')

plt.ylabel("Valor")
plt.title("Comparación de modelos por SSIM y PSNR")
plt.xticks(ticks=x, labels=modelos)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)

#GUARDAMOS LA FIGURA
plt.tight_layout()
plt.savefig("metrics/grafico_metricas.png")         #NO DEBO OLVIDAR CAMBIAR EL NOMBRE AQUÍ PARA EVITAR CONFLICTOS DE NOMBRE
plt.show()

#EXPLICACIÓN
mejor_modelo_ssim = mean_df.loc[mean_df["ssim"].idxmax(), "modelo"]
mejor_modelo_psnr = mean_df.loc[mean_df["psnr"].idxmax(), "modelo"]

print("\nAnálisis automático:")
print(f"→ El modelo con mayor similitud estructural (SSIM) es: **{mejor_modelo_ssim}**")
print(f"→ El modelo con menor pérdida de señal (PSNR) es: **{mejor_modelo_psnr}**")

if mejor_modelo_ssim == mejor_modelo_psnr:
    print(f"El modelo **{mejor_modelo_ssim}** es consistentemente superior según ambas métricas.")
else:
    print("Las métricas no coinciden: un modelo es mejor en estructura, otro en fidelidad de señal.")