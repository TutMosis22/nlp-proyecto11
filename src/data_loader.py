"""
Este módulo define funciones para cargar datos de entrada de texto, que serán usados para generar imágenes con modelos de IA generativa
"""

import os

def cargar_textos_desde_archivo(ruta_archivo):
    """
    Lee un archivo de texto línea por línea.
    
    Parámetros:
    ruta_archivo (str): Ruta al archivo de texto plano donde cada línea es una entrada.
    
    Retorna:
    listas_textos (list): Lista con las líneas de texto leídas.
    """
    
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"El archivo '{ruta_archivo}' no existe.")
    
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        lineas = f.readlines()
        
    #Elimina los saltos de línea y espacios extra
    lista_textos = [linea.strip() for linea in lineas if linea.strip()]
    
    return lista_textos

def ejemplo_textos():
    """
    Devuelve una lista de textos de ejemplo para pruebas
    """
    return [
        "A dog wearing sunglasses at the beach",
        "An astronaut riding a horse on Mars",
        "A beautiful landscape of mountains during sunset"
    ]