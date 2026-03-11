"""
comparar_avg.py — Por cada par de personas (A, B) calcula la distancia promedio
entre todas las fotos de A y todas las de B en el espacio ISOMAP. Ordena los pares
por esa distancia y muestra los 5 pares de personas más parecidos (menor score = más parecidos).
No devuelve fotos concretas, solo nombres y score.
"""

from pathlib import Path

import cv2
import numpy as np
from sklearn.manifold import Isomap
from sklearn.metrics.pairwise import euclidean_distances

CARAS_30X30_DIR = Path(__file__).resolve().parent / "caras_30x30"

ISOMAP_N_NEIGHBORS = 8
ISOMAP_N_COMPONENTS = 15


def imagen_a_vector(imagen_30x30: np.ndarray) -> np.ndarray:
    """Aplana una imagen 30x30 a un vector de 900 componentes."""
    return imagen_30x30.astype(np.float64).ravel()


def cargar_vectores_y_metadatos(ruta_base: Path) -> tuple[np.ndarray, list[str], list[Path]]:
    """
    Recorre caras_30x30/ por subcarpetas (persona).
    Devuelve X (n_muestras, 900), personas[i], paths[i].
    """
    if not ruta_base.is_dir():
        raise FileNotFoundError(f"No existe la carpeta '{ruta_base}'.")

    vectores: list[np.ndarray] = []
    personas: list[str] = []
    paths: list[Path] = []
    extensiones = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for persona_dir in sorted(ruta_base.iterdir()):
        if not persona_dir.is_dir():
            continue
        persona = persona_dir.name
        for path in sorted(persona_dir.iterdir()):
            if path.suffix.lower() not in extensiones:
                continue
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None or img.size == 0:
                continue
            if img.shape != (30, 30):
                img = cv2.resize(img, (30, 30))
            vectores.append(imagen_a_vector(img))
            personas.append(persona)
            paths.append(path)

    if not vectores:
        raise ValueError(f"No se encontraron imágenes en '{ruta_base}'.")

    X = np.vstack(vectores)
    return X, personas, paths


def aplicar_isomap(
    X: np.ndarray,
    n_components: int = ISOMAP_N_COMPONENTS,
    n_neighbors: int = ISOMAP_N_NEIGHBORS,
) -> np.ndarray:
    """ISOMAP sobre X (n_muestras, 900) → X_embed (n_muestras, n_components)."""
    n_samples = len(X)
    n_neighbors = min(n_neighbors, n_samples - 1)
    if n_neighbors < 2:
        return X
    n_comp = max(1, min(n_components, n_samples - 1, X.shape[1]))
    isomap = Isomap(n_components=n_comp, n_neighbors=n_neighbors)
    return isomap.fit_transform(X)


def top5_pares_distancia_promedio(
    X_embed: np.ndarray,
    personas: list[str],
) -> list[tuple[str, str, float]]:
    """
    Para cada par de personas (A, B) calcula la distancia promedio entre todas
    las fotos de A y todas las de B en X_embed. Devuelve los 5 pares con menor
    score (más parecidos). Cada elemento: (persona_a, persona_b, score).
    """
    personas_unicas = sorted(set(personas))
    if len(personas_unicas) < 2:
        raise ValueError("Se necesitan al menos dos personas para comparar.")

    resultados: list[tuple[str, str, float]] = []

    for i, a in enumerate(personas_unicas):
        for j, b in enumerate(personas_unicas):
            if i >= j:
                continue
            idx_a = [idx for idx, p in enumerate(personas) if p == a]
            idx_b = [idx for idx, p in enumerate(personas) if p == b]
            vec_a = X_embed[idx_a]
            vec_b = X_embed[idx_b]
            distancias = euclidean_distances(vec_a, vec_b)
            score = float(np.mean(distancias))
            resultados.append((a, b, score))

    resultados.sort(key=lambda x: x[2])
    return resultados[:5]


def main() -> None:
    print("Cargando caras desde", CARAS_30X30_DIR)
    X, personas, paths = cargar_vectores_y_metadatos(CARAS_30X30_DIR)
    n_personas = len(set(personas))
    print(f"Cargadas {len(personas)} imágenes de {n_personas} personas.")

    print("Aplicando ISOMAP...")
    X_embed = aplicar_isomap(X)

    print("Calculando distancia promedio entre cada par de personas...")
    top5 = top5_pares_distancia_promedio(X_embed, personas)

    print("\nTop 5 pares de personas más parecidos (menor score = más parecidos):")
    for i, (a, b, score) in enumerate(top5, 1):
        print(f"  {i}. '{a}' y '{b}' — score: {score:.4f}")


if __name__ == "__main__":
    main()
