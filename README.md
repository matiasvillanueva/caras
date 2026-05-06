# DMA – TP Caras + ISOMAP

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

1. Descargar el dataset desde SharePoint y descomprimirlo dentro de `caras/original/` (las fotos van sueltas, sin subcarpetas):

   <https://alumniiaeedu-my.sharepoint.com/shared?listurl=https%3A%2F%2Falumniiaeedu%2Dmy%2Esharepoint%2Ecom%2Fpersonal%2Fmceriotti%5Fmail%5Faustral%5Fedu%5Far%2FDocuments&id=%2Fpersonal%2Fmceriotti%5Fmail%5Faustral%5Fedu%5Far%2FDocuments%2FCaras&viewid=71522521%2Dbbec%2D4825%2D9802%2D8b3b07729c50>

2. (Opcional) Para test manual, dropear fotos crudas en `caras/original_custom_test/<persona>/`. Para personas no conocidas, usar `caras/original_custom_test/otros/`.

3. Ejecutar el pipeline:

   ```bash
   cd caras
   python procesar.py
   ```

4. Entrenar y evaluar la red:

   ```bash
   python examen.py
   ```

## Estructura

```
caras/
├── procesar.py
├── examen.py
├── config.py
├── steps/
│   ├── recortar_y_partir_train_test.py
│   ├── recortar_original_custom_test.py
│   ├── entrenar_isomap_y_exportar_txt.py
│   └── reconstruir_caras_isomap.py
├── utils/
│   ├── extraer_nombre_persona.py
│   ├── convertir_heic_a_jpeg.py
│   ├── listar_imagenes_originales.py
│   └── recorte_rostro_dos_ojos.py
├── original/                       # fotos crudas (descargadas de SharePoint)
├── original_custom_test/           # fotos crudas para test manual
│   ├── <persona>/
│   └── otros/
├── caras_1200/                     # generado: recortes 1200x1200
│   ├── training/<persona>/
│   ├── test/<persona>/
│   └── custom_test/<persona>/
└── isomap/                         # generado: embeddings 70D y reconstrucciones
    ├── training_embeddings.txt
    ├── test_embeddings.txt
    ├── test_custom_embeddings.txt
    ├── training/<persona>/
    ├── test/<persona>/
    └── custom_test/<persona>/
```
