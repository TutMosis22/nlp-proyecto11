from flask import Flask, render_template, request, redirect, url_for
import os
import random
import csv

app = Flask(__name__)
RESULTS_CSV = "metrics/resultados_humanos.csv"

# Puedes modificar aquí el nombre de las imágenes que deseas mostrar
IMAGENES = [
    ("openai", "static/openai_dalle_output.png"),
    ("transformers", "static/transformers_image_1.png"),
    ("diffusers", "static/diffusers_output.png"),
]
PROMPT = "A futuristic cityscape at sunset"

@app.route("/", methods=["GET", "POST"])
def evaluar():
    if request.method == "POST":
        seleccion = request.form.get("mejor_imagen")
        modelo_seleccionado = request.form.get("modelo")

        with open(RESULTS_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([PROMPT, modelo_seleccionado, seleccion])

        return redirect(url_for("gracias"))

    # Se baraja el orden de las imágenes para evitar sesgos
    imagenes_barajadas = IMAGENES.copy()
    random.shuffle(imagenes_barajadas)

    return render_template("formulario.html", 
                           imagen_base="static/cityscape.png",
                           imagenes=imagenes_barajadas)

@app.route("/gracias")
def gracias():
    return "<h2>¡Gracias por tu evaluación!</h2>"

if __name__ == "__main__":
    os.makedirs("metrics", exist_ok=True)
    app.run(debug=True)
