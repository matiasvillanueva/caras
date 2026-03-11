"""Procesamiento de imágenes: lee original/, detecta caras frontales, guarda 30x30 en gris en caras_30x30/."""

import re
from pathlib import Path

import cv2
import numpy as np

import pillow_heif
from PIL import Image

ORIGINAL_DIR = "original"
CARAS_30X30_DIR = "caras_30x30"

# Cascades de OpenCV (incluidos en el paquete)
CASCADE_FACE = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
CASCADE_EYES = Path(cv2.data.haarcascades) / "haarcascade_eye.xml"

# Correcciones de nombres (typos): clave -> nombre canónico
NOMBRE_CANONICO: dict[str, str] = {"migue": "miguel", "juan": "juani"}

def extraer_nombre_persona(nombre_archivo: str) -> str:
    """
    Extrae solo el nombre (sin apellido): primera parte antes del primer '_', en minúsculas.
    Ej: matias_villanueva (20).jpeg -> matias, Lucia_Tamplin_03.png -> lucia.
    Aplica correcciones hardcodeadas (ej. migue -> miguel).
    """
    stem = Path(nombre_archivo).stem
    sin_parentesis = re.sub(r"\s*\([^)]*\)", "", stem)
    sin_numeros = re.sub(r"\d+$", "", sin_parentesis).rstrip("_ -")
    nombre = sin_numeros.split("_")[0].strip().lower()
    return NOMBRE_CANONICO.get(nombre, nombre)


def _cargar_imagen(path: Path) -> np.ndarray | None:
    """Carga una imagen desde path. Soporta formatos de OpenCV y .heic (vía pillow-heif)."""
    suf = path.suffix.lower()
    if suf == ".heic":
        pillow_heif.register_heif_opener()
        pil_img = Image.open(path)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        arr = np.array(pil_img)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    img = cv2.imread(str(path))
    return img


def _es_cara_frontal(imagen_gris: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
    """
    Valida que la cara sea frontal: al menos 2 ojos detectados dentro del rostro,
    a altura similar (excluye perfiles y falsos positivos). Se llama sobre la imagen
    completa en gris, antes de recortar.
    """
    eye_cascade = cv2.CascadeClassifier(str(CASCADE_EYES))
    ojos = eye_cascade.detectMultiScale(imagen_gris, scaleFactor=1.1, minNeighbors=4, minSize=(10, 10))
    x2 = x + w
    zona_ojos_y = y + int(h * 0.15)
    zona_ojos_h = int(h * 0.55)
    dentro = []
    for (ox, oy, ow, oh) in ojos:
        cx = ox + ow // 2
        cy = oy + oh // 2
        if x <= cx <= x2 and zona_ojos_y <= cy <= zona_ojos_y + zona_ojos_h:
            dentro.append(cy)
    if len(dentro) < 2:
        return False
    dentro.sort()
    if abs(dentro[-1] - dentro[0]) > h * 0.35:
        return False
    return True


def detectar_y_recortar_cara(imagen: np.ndarray) -> np.ndarray | None:
    """
    Detecta una cara con OpenCV en la imagen (color); valida que sea frontal (2 ojos);
    recorta desde la imagen en color y devuelve el recorte en escala de grises.
    """
    if imagen is None or imagen.size == 0:
        return None
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) if len(imagen.shape) == 3 else imagen
    face_cascade = cv2.CascadeClassifier(str(CASCADE_FACE))
    caras = face_cascade.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(caras) == 0:
        return None
    (x, y, w, h) = max(caras, key=lambda r: r[2] * r[3])
    if not _es_cara_frontal(gris, x, y, w, h):
        return None
    recorte_color = imagen[y : y + h, x : x + w]
    recorte_gris = cv2.cvtColor(recorte_color, cv2.COLOR_BGR2GRAY)
    return recorte_gris


def preprocesar() -> None:
    """
    Lee todas las imágenes desde original/, detecta cara frontal con OpenCV,
    guarda solo la versión 30x30 en escala de grises en caras_30x30/<persona>/.
    """
    original = Path(ORIGINAL_DIR)
    if not original.is_dir():
        raise FileNotFoundError(f"No existe la carpeta '{ORIGINAL_DIR}'. Colocá ahí las fotos.")

    caras_30 = Path(CARAS_30X30_DIR)
    caras_30.mkdir(exist_ok=True)

    extensiones = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic"}
    procesadas = 0
    sin_cara = 0
    contador_por_persona: dict[str, int] = {}

    for path in sorted(original.iterdir()):
        if path.suffix.lower() not in extensiones:
            continue
        persona = extraer_nombre_persona(path.name)
        if not persona:
            continue
        (caras_30 / persona).mkdir(exist_ok=True)

        img = _cargar_imagen(path)
        recorte = detectar_y_recortar_cara(img)
        if recorte is None:
            sin_cara += 1
            continue
        contador_por_persona[persona] = contador_por_persona.get(persona, 0) + 1
        num = contador_por_persona[persona]
        nombre_archivo = f"{persona}_{num:03d}.png"
        pequena = cv2.resize(recorte, (30, 30))
        out_30 = caras_30 / persona / nombre_archivo
        cv2.imwrite(str(out_30), pequena)
        procesadas += 1

    print(f"Preprocesamiento: {procesadas} imágenes guardadas en {CARAS_30X30_DIR}/.")
    if sin_cara:
        print(f"  {sin_cara} imagen(es) sin cara frontal omitidas.")


if __name__ == "__main__":
    preprocesar()
