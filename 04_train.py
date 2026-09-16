"""
Paso 4: Entrenar el modelo de detección de un solo paso (YOLOv8n) mediante
transfer learning, tal como plantea el Objetivo específico 2 de la tesis.

Configuración ajustada para Asus TUF Gaming (GTX 1650, 4GB VRAM / 8GB RAM):
  - imgsz=416: reduce el uso de memoria de video vs. el estándar 640.
  - batch=16: cabe cómodo en 4GB con YOLOv8n a 416px. Si ves un error de
    "CUDA out of memory", baja a batch=8 o batch=4.
  - cache=False: con solo 8GB de RAM, cachear todo el dataset en memoria
    puede saturarla. Déjalo en False (lee desde disco).
  - amp=True (por defecto en Ultralytics): usa precisión mixta (FP16),
    lo que reduce memoria y acelera el entrenamiento en la GTX 1650.

Métricas objetivo según la tesis: precisión > 85%, falsos positivos < 5%.
"""

import torch
from ultralytics import YOLO

DATA_YAML = "configs/data.yaml"


def get_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"GPU detectada: {name}. Se usará device=0 (CUDA).")
        return 0
    print("No se detectó GPU con CUDA. Se entrenará en CPU (más lento).")
    print("Revisa el requirements.txt: instala torch con soporte CUDA 11.8.")
    return "cpu"


def main():
    device = get_device()

    # Pesos preentrenados en COCO -> transfer learning
    model = YOLO("yolov8n.pt")

    model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=416,
        batch=16,             # bajar a 8 o 4 si aparece "CUDA out of memory"
        device=device,
        workers=4,             # bajar a 2 si el PC se congela cargando datos
        cache=False,           # no cachear en RAM (solo 8GB disponibles)
        patience=20,           # early stopping si no mejora en 20 épocas
        project="runs_detect",
        name="intrusos_nir_v1",
        augment=True,
        amp=True,               # precisión mixta: menos VRAM, más velocidad
        verbose=True,
    )

    # Validación final con métricas (precision, recall, mAP)
    metrics = model.val()
    print(metrics)

    print("\nPesos finales guardados en: runs_detect/intrusos_nir_v1/weights/best.pt")


if __name__ == "__main__":
    main()
