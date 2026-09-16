"""
Paso 2: Simular una cámara de visión nocturna NIR (infrarrojo cercano)
a partir de imágenes RGB reales.

Esta es la implementación concreta del "entorno de simulación de cámaras"
descrito en la tesis (Limitación 1 y Objetivo específico 1): en vez de
capturar con hardware físico, transformamos imágenes diurnas/RGB en
imágenes que emulan lo que vería un sensor NIR bajo iluminación IR activa:

  - Escala de grises (el NIR no captura color)
  - Ganancia + gamma para simular el realce de contraste típico de estos
    sensores
  - Viñeteado radial: simula la caída de intensidad del foco IR a medida
    que un objeto se aleja del centro/alcance del iluminador
  - Ruido gaussiano: simula el ruido de sensor característico de cámaras
    NIR de bajo costo en condiciones de poca luz

Los parámetros (ir_gain, noise_std, vignette_strength) son justamente los
"parámetros configurados de potencia y alcance del foco IR" que mencionas
en la Limitación 1 del documento: puedes variarlos para generar variantes
con distinta "distancia de iluminación simulada", lo cual también sirve
como data augmentation.
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm

RAW_DIR = "data/raw_openimages/data"
OUT_DIR = "data/nir_simulated"


def simulate_nir(img_bgr, ir_gain=1.15, gamma=0.8, noise_std=6.0, vignette_strength=0.35):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Ganancia simulando sensibilidad del sensor NIR
    gray = np.clip(gray * ir_gain, 0, 255)

    # Corrección gamma: realza contraste como en sensores IR reales
    gray = np.power(gray / 255.0, gamma) * 255.0

    # Viñeteado radial: simula el alcance limitado del foco IR
    h, w = gray.shape
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2.0, h / 2.0
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    vignette = 1 - vignette_strength * (dist / max_dist)
    gray = gray * vignette

    # Ruido de sensor de bajo costo
    noise = np.random.normal(0, noise_std, gray.shape)
    gray = np.clip(gray + noise, 0, 255).astype(np.uint8)

    return gray


def process_all(src_dir=RAW_DIR, dst_dir=OUT_DIR, variants_per_image=1):
    """
    variants_per_image > 1 genera varias versiones NIR (distintos parámetros
    de iluminación simulada) por cada foto original, útil como data
    augmentation adicional al de YOLO.
    """
    os.makedirs(dst_dir, exist_ok=True)
    files = list(Path(src_dir).glob("*.jpg")) + list(Path(src_dir).glob("*.png"))

    if not files:
        print(f"No se encontraron imágenes en {src_dir}. "
              f"¿Corriste primero 01_download_dataset.py?")
        return

    for f in tqdm(files, desc="Simulando NIR"):
        img = cv2.imread(str(f))
        if img is None:
            continue

        for v in range(variants_per_image):
            gain = np.random.uniform(0.9, 1.3)
            vignette = np.random.uniform(0.2, 0.45)
            noise = np.random.uniform(3, 10)
            nir = simulate_nir(img, ir_gain=gain, noise_std=noise, vignette_strength=vignette)

            suffix = f"_v{v}" if variants_per_image > 1 else ""
            out_name = f"{f.stem}{suffix}.jpg"
            cv2.imwrite(str(Path(dst_dir) / out_name), nir)

    print(f"Listo. Imágenes NIR simuladas guardadas en: {dst_dir}")


if __name__ == "__main__":
    process_all()
