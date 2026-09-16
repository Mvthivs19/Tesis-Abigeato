"""
Paso 3: Convertir las anotaciones (formato COCO, generadas en el paso 1)
al formato que espera YOLO, y dividir el dataset en train/val/test.

Nota: por defecto, el paso 2 (02_simulate_nir.py) genera UNA imagen NIR
por cada imagen original, con el MISMO nombre de archivo, así que las
cajas (bounding boxes) originales de Open Images siguen siendo válidas
sin ningún cambio geométrico (las transformaciones NIR no mueven píxeles,
solo cambian tono/ruido/viñeteado).

Estructura final generada:
  data/yolo_dataset/
    images/train/*.jpg
    images/val/*.jpg
    images/test/*.jpg
    labels/train/*.txt
    labels/val/*.txt
    labels/test/*.txt
  configs/data.yaml
"""

import json
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml

COCO_JSON = "data/raw_openimages/labels.json"
NIR_IMAGES_DIR = "data/nir_simulated"
OUT_DIR = "data/yolo_dataset"
DATA_YAML_PATH = "configs/data.yaml"

# Debe coincidir EXACTAMENTE con el orden de CLASSES en 01_download_dataset.py
CLASS_NAMES = ["humano", "bovino", "equino", "ovino", "porcino"]
OPENIMAGES_TO_LOCAL = {
    "Person": "humano",
    "Cattle": "bovino",
    "Horse": "equino",
    "Sheep": "ovino",
    "Pig": "porcino",
}


def load_coco(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def coco_to_yolo_bbox(bbox, img_w, img_h):
    # COCO: [x_min, y_min, width, height] en píxeles absolutos
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    return x_center, y_center, w / img_w, h / img_h


def main():
    coco = load_coco(COCO_JSON)

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    images_by_id = {img["id"]: img for img in coco["images"]}

    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}

    # Genera pares (nombre_archivo, lista_de_labels_yolo)
    records = []
    for img_id, img_info in images_by_id.items():
        file_name = img_info["file_name"]
        nir_path = Path(NIR_IMAGES_DIR) / file_name
        if not nir_path.exists():
            continue  # imagen no simulada (o corrupta en paso 2)

        w, h = img_info["width"], img_info["height"]
        anns = anns_by_image.get(img_id, [])
        if not anns:
            continue

        yolo_lines = []
        for ann in anns:
            coco_name = cat_id_to_name.get(ann["category_id"])
            local_name = OPENIMAGES_TO_LOCAL.get(coco_name)
            if local_name is None:
                continue
            cls_idx = class_to_idx[local_name]
            xc, yc, bw, bh = coco_to_yolo_bbox(ann["bbox"], w, h)
            yolo_lines.append(f"{cls_idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        if yolo_lines:
            records.append((str(nir_path), file_name, yolo_lines))

    print(f"Total de imágenes con anotaciones válidas: {len(records)}")

    # División 80/10/10
    train_recs, temp_recs = train_test_split(records, test_size=0.2, random_state=42)
    val_recs, test_recs = train_test_split(temp_recs, test_size=0.5, random_state=42)

    splits = {"train": train_recs, "val": val_recs, "test": test_recs}

    for split_name, recs in splits.items():
        img_out = Path(OUT_DIR) / "images" / split_name
        lbl_out = Path(OUT_DIR) / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for src_path, file_name, yolo_lines in recs:
            shutil.copy(src_path, img_out / file_name)
            txt_name = Path(file_name).stem + ".txt"
            with open(lbl_out / txt_name, "w") as f:
                f.write("\n".join(yolo_lines))

        print(f"  {split_name}: {len(recs)} imágenes")

    # Genera configs/data.yaml para Ultralytics
    os.makedirs(Path(DATA_YAML_PATH).parent, exist_ok=True)
    data_yaml = {
        "path": os.path.abspath(OUT_DIR),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    with open(DATA_YAML_PATH, "w") as f:
        yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    print(f"\nListo. Configuración generada en: {DATA_YAML_PATH}")


if __name__ == "__main__":
    main()
