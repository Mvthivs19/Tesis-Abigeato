"""
Consolida todos los datasets en uno solo para entrenamiento YOLO.
Combina: Open Images (person, cattle, horse) + Ovejas + Cerdos
"""

import json
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

OUT_DIR = Path("data/custom_yolo")
RAW_DIR = Path("data/raw_openimages")
CLASSES = ["humano", "bovino", "equino", "ovino", "porcino"]
OPENIMAGES_MAP = {"Person": "humano", "Cattle": "bovino", "Horse": "equino", "Sheep": "ovino", "Pig": "porcino"}
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}


def clean_dirs():
    if OUT_DIR.exists():
        shutil.rmtree(str(OUT_DIR))
    for split in ["train", "val", "test"]:
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def process_open_images():
    coco_path = RAW_DIR / "labels.json"
    if not coco_path.exists():
        print("No se encontro labels.json de Open Images")
        return []

    coco = json.loads(coco_path.read_text())
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    img_info = {img["id"]: img for img in coco["images"]}

    anns_by_img = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    records = []
    for img_id, info in img_info.items():
        open_name = cats.get(anns_by_img.get(img_id, [{}])[0].get("category_id", -1), "")
        local_name = OPENIMAGES_MAP.get(open_name)
        if not local_name:
            continue

        img_path = RAW_DIR / "data" / info["file_name"]
        if not img_path.exists():
            continue

        w, h = info["width"], info["height"]
        yolo_lines = []
        for ann in anns_by_img.get(img_id, []):
            cls_name = OPENIMAGES_MAP.get(cats.get(ann["category_id"], ""), "")
            if cls_name is None:
                continue
            cls_idx = CLASS_TO_IDX[cls_name]
            x, y, bw, bh = ann["bbox"]
            xc = (x + bw / 2) / w
            yc = (y + bh / 2) / h
            yolo_lines.append(f"{cls_idx} {xc:.6f} {yc:.6f} {bw/w:.6f} {bh/h:.6f}")

        if yolo_lines:
            records.append((str(img_path), info["file_name"], "\n".join(yolo_lines)))

    print(f"Open Images: {len(records)} imagenes con anotaciones")
    return records


def process_ovejas():
    img_dir = Path("C:/Users/mabad/Desktop/Ovejas/images")
    lbl_dir = Path("C:/Users/mabad/Desktop/Ovejas/annotations")

    import xml.etree.ElementTree as ET

    records = []
    for xml_path in lbl_dir.glob("*.xml"):
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
        size = root.find("size")
        if size is None:
            continue
        w = int(size.find("width").text)
        h = int(size.find("height").text)

        img_name = xml_path.stem + ".png"
        img_path = img_dir / img_name
        if not img_path.exists():
            continue

        yolo_lines = []
        for obj in root.findall("object"):
            name = obj.find("name").text.lower().strip()
            if name == "sheep":
                cls_idx = CLASS_TO_IDX["ovino"]
                bbox = obj.find("bndbox")
                xmin = int(bbox.find("xmin").text)
                ymin = int(bbox.find("ymin").text)
                xmax = int(bbox.find("xmax").text)
                ymax = int(bbox.find("ymax").text)
                xc = (xmin + xmax) / 2.0 / w
                yc = (ymin + ymax) / 2.0 / h
                bw = (xmax - xmin) / w
                bh = (ymax - ymin) / h
                yolo_lines.append(f"{cls_idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        if yolo_lines:
            records.append((str(img_path), img_name, "\n".join(yolo_lines)))

    print(f"Ovejas: {len(records)} imagenes con anotaciones")
    return records


def process_cerdos():
    img_dir = Path("C:/Users/mabad/Desktop/Cerdos/pig/train/images")
    lbl_dir = Path("C:/Users/mabad/Desktop/Cerdos/pig/train/labels")

    records = []
    for lbl_path in lbl_dir.glob("*.txt"):
        img_path = img_dir / (lbl_path.stem + ".jpg")
        if not img_path.exists():
            img_path = img_dir / (lbl_path.stem + ".png")
        if not img_path.exists():
            continue

        content = lbl_path.read_text().strip()
        if content:
            lines = []
            for line in content.splitlines():
                parts = line.strip().split()
                if parts:
                    lines.append(f"4 {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
            if lines:
                records.append((str(img_path), img_path.name, "\n".join(lines)))

    print(f"Cerdos: {len(records)} imagenes con anotaciones")
    return records


def process_caballos():
    src_dir = Path("C:/Users/mabad/Desktop/Caballos")
    records = []
    for img_path in list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png")):
        records.append((str(img_path), img_path.name, ""))

    print(f"Caballos: {len(records)} imagenes (sin anotaciones)")
    return records


def save_split(records, split_name):
    img_out = OUT_DIR / "images" / split_name
    lbl_out = OUT_DIR / "labels" / split_name

    for src_path, file_name, yolo_lines in records:
        shutil.copy2(src_path, str(img_out / file_name))
        txt_name = Path(file_name).stem + ".txt"
        with open(lbl_out / txt_name, "w") as f:
            f.write(yolo_lines)

    return len(records)


def generate_data_yaml():
    data_yaml = {
        "path": str(OUT_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(CLASSES)},
    }
    with open("configs/custom_data.yaml", "w") as f:
        import yaml
        yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)
    print("configs/custom_data.yaml generado")


if __name__ == "__main__":
    print("=== Consolidando datasets ===\n")

    clean_dirs()

    all_records = []
    all_records.extend(process_open_images())
    all_records.extend(process_ovejas())
    all_records.extend(process_cerdos())
    all_records.extend(process_caballos())

    print(f"\nTotal: {len(all_records)} imagenes")

    train_recs, temp_recs = train_test_split(all_records, test_size=0.2, random_state=42)
    val_recs, test_recs = train_test_split(temp_recs, test_size=0.5, random_state=42)

    n_train = save_split(train_recs, "train")
    n_val = save_split(val_recs, "val")
    n_test = save_split(test_recs, "test")

    print(f"\nDivision: train={n_train}, val={n_val}, test={n_test}")

    generate_data_yaml()

    print("\nDataset listo para entrenar:")
    print("  python src/04_train_custom.py")
