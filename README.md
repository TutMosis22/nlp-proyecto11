# Proyecto 11: Explorando librerías de IA generativa y creación de un pipeline modular

## 🎯 Objetivo
Este proyecto tiene como finalidad evaluar y comparar diversas librerías de IA generativa aplicadas a texto e imagen, integrando un pipeline modular capaz de combinar distintos modelos, tokenizers y estrategias de prompting. Además, se medirán métricas de calidad tanto automáticas como humanas.

## 📦 Librerías y herramientas exploradas
- **Transformers (Hugging Face)**: Modelos preentrenados, decodificación con sampling (`top-k`, `nucleus`, `beam`).
- **OpenAI API**: Acceso a modelos tipo `gpt-*` vía REST API.
- **Diffusers**: Generación de imágenes con `StableDiffusionPipeline`.
- **Métricas**: BLEU, ROUGE-L, METEOR, Self-BLEU, SSIM y PSNR.

## 📁 Estructura del repositorio
- **src/**  Código fuente organizado por componentes 
- **notebooks/**  Cuadernos de prueba, métricas y visualizaciones 
- **metrics/**  Métricas y resultados numéricos o gráficos 
- **docs/**  Documentación y presentación final


## 🧪 Cómo ejecutar
### Requisitos
- Python
- pip + virtualenv
- Acceso a Google Colab o entorno local

### Ejecución local
```bash
# Clona este repositorio
git clone https://github.com/TutMosis22/nlp-proyecto11.git
cd nlp-proyecto11

# Crea y activa un entorno virtual
python -m venv venv
source venv/bin/activate   # En Windows usa: .\venv\Scripts\activate

# Instala dependencias
pip install -r requirements.txt
