# DMA – TP Caras + ISOMAP

Pipeline: fotos en `original/` → caras 30×30 en gris en `caras_30x30/` → comparación con ISOMAP para hallar parecidos entre personas.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

1. Colocá las fotos (una cara por imagen) en la carpeta **`original/`**. Se usa solo el nombre (primera parte antes de `_`), en minúsculas (ej. `Juan_Perez_01.jpg` → carpeta `juan/`).

2. **Procesamiento** (detecta caras frontales con OpenCV y guarda 30×30 en gris en `caras_30x30/`):
   ```bash
   python procesar.py
   ```

3. **Comparación** — hay dos scripts que leen `caras_30x30/`, construyen vectores (900 dim) y aplican ISOMAP:

   - **`comparar_avg.py`** — Por cada par de personas (A, B) calcula la **distancia promedio** entre todas las fotos de A y todas las de B en el espacio ISOMAP. Ordena los pares por esa distancia y muestra los 5 pares de personas más parecidos (menor score = más parecidos). No devuelve fotos concretas, solo nombres y score.
   ```bash
   python comparar_avg.py
   ```

   - **`comparar_top.py`** — Por cada par de personas (A, B) busca el **par de fotos** (una de A, una de B) con **menor distancia** entre sí en el espacio ISOMAP. Ordena los pares de personas por esa distancia mínima y muestra los 5 pares con la foto más cercana, imprimiendo los paths de esas dos fotos.
   ```bash
   python comparar_top.py
   ```

Resumen: `comparar_avg` usa promedios entre todas las fotos de cada persona; `comparar_top` usa la pareja de fotos más cercana entre dos personas y muestra sus paths.
