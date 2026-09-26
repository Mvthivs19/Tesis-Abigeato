"""
Utilidad opcional: si no tienes un video propio para probar el pipeline
en tiempo real (paso 6), este script arma un video de prueba (mp4)
juntando las imágenes NIR simuladas del paso 2, para que puedas correr
todo el sistema de punta a punta sin depender de una cámara física.

No reemplaza tu dataset de entrenamiento; es solo para PROBAR que el
pipeline de inferencia + alertas + grabación funciona end-to-end.
"""

import cv2
from pathlib import Path

SRC_DIR = "data/nir_simulated"
OUT_PATH = "data/sample_video.mp4"
FPS = 10
MAX_FRAMES = 300  # ~30 segundos a 10 fps


def main():
    files = sorted(Path(SRC_DIR).glob("*.jpg"))[:MAX_FRAMES]
    if not files:
        print(f"No hay imágenes en {SRC_DIR}. Corre primero 02_simulate_nir.py.")
        return

    first = cv2.imread(str(files[0]))
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT_PATH, fourcc, FPS, (w, h), isColor=False)

    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (w, h))
        writer.write(img)

    writer.release()
    print(f"Video de prueba generado en: {OUT_PATH} ({len(files)} frames)")


if __name__ == "__main__":
    main()
