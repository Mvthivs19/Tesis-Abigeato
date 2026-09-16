"""
Paso 1: Descargar imágenes públicas desde Open Images V7.

Objetivo (tesis, obj. específico 1): obtener la materia prima para construir
el dataset de las 5 clases objetivo: Humano, Bovino, Equino, Ovino, Porcino.

Usamos la librería `fiftyone`, que descarga solo las imágenes y anotaciones
de las clases que pidamos (no el dataset completo, que pesa varios GB).

Ajusta SAMPLES_PER_CLASS según tu PC/conexión. Con 300-500 por clase
alcanza para un primer entrenamiento de prueba (proof of concept).
"""

import fiftyone as fo
import fiftyone.zoo as foz

# Mapeo clase Open Images -> clase de la tesis
CLASSES = ["Person", "Cattle", "Horse", "Sheep", "Pig"]
SAMPLES_PER_CLASS = 400  # subir a 1000+ más adelante si el resultado es pobre

OUTPUT_DIR = "data/raw_openimages"


def main():
    print(f"Descargando ~{SAMPLES_PER_CLASS} imágenes por clase: {CLASSES}")

    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["detections"],
        classes=CLASSES,
        max_samples=SAMPLES_PER_CLASS * len(CLASSES),
        seed=42,
        shuffle=True,
    )

    # Exportamos a formato COCO, que es fácil de convertir a YOLO después
    dataset.export(
        export_dir=OUTPUT_DIR,
        dataset_type=fo.types.COCODetectionDataset,
        classes=CLASSES,
    )

    print(f"Listo. Dataset crudo exportado en: {OUTPUT_DIR}")
    print("Estructura esperada: data/raw_openimages/data/*.jpg")
    print("                      data/raw_openimages/labels.json")


if __name__ == "__main__":
    main()
