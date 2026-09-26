"""
Paso 4: Entrenar el modelo de detección de un solo paso (YOLOv8s) mediante
transfer learning, tal como plantea el Objetivo específico 2 de la tesis.

Configuración óptima para Asus TUF Gaming (GTX 1650, 4GB VRAM / 8GB RAM):
  - YOLOv8s (small): 11.2M params, ~5-10% mejor precisión que YOLOv8n.
  - imgsz=640: resolución completa para mejor detección.
  - batch=8: seguro para 4GB VRAM con YOLOv8s.
  - cache=False: con 8GB RAM, no cachear en memoria.
  - amp=True: precisión mixta (FP16), reduce VRAM.
  - workers=2: carga paralela de imágenes (con freeze_support para Windows).

Dataset: 24,309 train / 2,446 val / 1,780 test (5 clases balanceadas)
Métricas objetivo: Precision > 85%, Recall > 70%, mAP50 > 70%.

PARA PAUSAR Y REANUDAR:
  1. Ctrl+C para pausar (se guarda last.pt automáticamente)
  2. Cambiar RESUME = True abajo
  3. Volver a ejecutar: python src/04_train.py
  4. Cuando termine, volver RESUME = False
"""

import os

# ==============================
# CAMBIAR A True PARA REANUDAR
# ==============================
RESUME = False
# ==============================

import torch

_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

from ultralytics import YOLO

DATA_YAML = "configs/custom_data.yaml"
RUN_NAME = "intrusos_custom_v15"


def get_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"GPU detectada: {name}. Se usará device=0 (CUDA).")
        return 0
    print("No se detectó GPU con CUDA. Se entrenará en CPU (más lento).")
    return "cpu"


def main():
    device = get_device()
    run_dir = os.path.join("runs_detect", RUN_NAME)
    weights_path = os.path.join(run_dir, "weights", "last.pt")

    if RESUME and os.path.exists(weights_path):
        print(f"REANUDANDO desde: {weights_path}")
        model = YOLO(weights_path)
    else:
        if RESUME:
            print("No se encontró checkpoint. Iniciando desde cero.")
        model = YOLO("yolov8s.pt")

    print("=" * 60)
    print("CONFIGURACIÓN DE ENTRENAMIENTO")
    print("=" * 60)
    print(f"Modelo:      YOLOv8s (11.2M params)")
    print(f"Épocas:      40")
    print(f"Resolución:  640px")
    print(f"Batch size:  8")
    print(f"Dataset:     24,309 train / 2,446 val")
    print(f"Dispositivo: {device}")
    print(f"Reanudar:    {RESUME}")
    print("=" * 60)

    model.train(
        data=DATA_YAML,
        epochs=40,
        imgsz=640,
        batch=8,
        device=device,
        workers=2,
        cache=False,
        patience=20,
        save_period=10,
        project="runs_detect",
        name=RUN_NAME,
        resume=RESUME,
        augment=True,
        amp=True,
        verbose=True,
        optimizer="auto",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
    )

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"Resultados en: runs_detect/{RUN_NAME}/")
    print("  - results.png")
    print("  - confusion_matrix.png")
    print("  - weights/best.pt")
    print("  - weights/last.pt")
    print("=" * 60)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
