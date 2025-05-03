from flask import Flask, render_template, request, redirect, url_for
import os
import csv
import random

app = Flask(__name__)
RESULTS_CSV = "metrics/resultados_humanos.csv"

# Puedes modificar aquí el nombre de las imágenes que deseas mostrar
IMAGENES = [
    ("openai", "openai_cityscape.png"),
    ("transformers", "transformers_cityscape.png"),
    ("diffusers", "diffusers_cityscape.png"),
]

@app.route("/", methods=["GET", "POST"])
def evaluar():
    metricas = ["realismo", "correspondencia", "calidad", "preferencia"]
    prompt = "cityscape"

    if request.method == "POST":
        fila = [prompt]
        for modelo, _ in IMAGENES:
            for metrica in metricas:
                valor = request.form.get(f"{modelo}_{metrica}")
                fila.append(valor)
               
        escribir_encabezados = not os.path.exists(RESULTS_CSV) or os.stat(RESULTS_CSV).st_size == 0        
                
        with open(RESULTS_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            if escribir_encabezados:
                encabezados = (["prompt"] + [
                    f"{modelo}_{metrica}"
                    for modelo in IMAGENES
                    for metrica in metricas
                ])
                writer.writerow(encabezados)
            writer.writerow(fila)
        return redirect(url_for("gracias"))

    imagen_base = "cityscape.png"
    imagenes_modelos = IMAGENES.copy()
#    random.shuffle(imagenes_modelos)

    return render_template("formulario.html",
                           imagen_base=imagen_base,
                           imagenes=imagenes_modelos)

@app.route("/gracias")
def gracias():
    return "<h2>¡Gracias por tu evaluación!</h2>"

if __name__ == "__main__":
    os.makedirs("metrics", exist_ok=True)
    app.run(debug=True)
