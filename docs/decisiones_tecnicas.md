# Registro de Decisiones Técnicas

## Modelo Generativo

- Se eligió `StableDiffusionPipeline` de `diffusers` por su alta calidad y facilidad de uso.
- Se descartó usar modelos más grandes como DALL·E 2 por requerimientos de cómputo, que aunque lo poseo (de cierta forma), no es lo mejor por fines prácticos.

## Estructura del Proyecto

- Se modularizó en `my_pipeline.py` para facilitar la reutilización de código.
- Se creó carpeta `notebooks/` para pruebas exploratorias.
- Se añadió `metrics/` para guardar resultados de pruebas, y `docs/` para documentar decisiones.

## Alternativas Consideradas

- Intento fallido de usar la librería `pipeline` generó conflicto de nombre. Se resolvió renombrando a `my_pipeline.py`.
