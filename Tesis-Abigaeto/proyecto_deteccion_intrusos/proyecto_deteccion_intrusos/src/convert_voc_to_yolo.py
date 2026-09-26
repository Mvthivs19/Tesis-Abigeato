"""
Convierte anotaciones Pascal VOC (XML) a formato YOLO (txt).
Uso: python src/convert_voc_to_yolo.py

Busca imágenes en data/custom_dataset/raw/ y genera las anotaciones
en data/custom_dataset/labels/.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

RAW_DIRS = {
    "ovino": "C:/Users/mabad/Desktop/Ovejas",
    "bovino": "C:/Users/mabad/Desktop/Vacas",
    "equino": "C:/Users/mabad/Desktop/Caballos",
    "porcino": "C:/Users/mabad/Desktop/Cerdos",
}

OUT_DIR = "data/custom_dataset"
CLASS_NAMES = ["humano", "bovino", "equino", "ovino", "porcino"]
CLASS_MAP_VOC = {
    "sheep": "ovino",
    "cow": "bovino",
    "cattle": "bovino",
    "horse": "equino",
    "pig": "porcino",
    "person": "humano",
}


def voc_to_yolo_bbox(bbox, img_w, img_h):
    xmin = int(bbox.find("xmin").text)
    ymin = int(bbox.find("ymin").text)
    xmax = int(bbox.find("xmax").text)
    ymax = int(bbox.find("ymax").text)
    x_center = (xmin + xmax) / 2.0 / img_w
    y_center = (ymin + ymax) / 2.0 / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    return x_center, y_center, width, height


def convert_xml(xml_path, class_map):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("img") if root.find("img") is not None else root.find("size")
    if size is None:
        return None
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    yolo_lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text.lower().strip()
        if name in class_map:
            cls_name = class_map[name]
            cls_idx = CLASS_NAMES.index(cls_name)
            bbox = obj.find("bndbox")
            if bbox is not None:
                xc, yc, w, h = voc_to_yolo_bbox(bbox, img_w, img_h)
                yolo_lines.append(f"{cls_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return yolo_lines


def copy_yolo_format(src_img_dir, src_label_dir, dst_img_dir, dst_label_dir, extension="*.jpg"):
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_label_dir, exist_ok=True)

    images = list(Path(src_img_dir).glob(extension)) + list(Path(src_img_dir).glob("*.png"))
    count = 0
    for img_path in images:
        lbl_path = Path(src_label_dir) / (img_path.stem + ".txt")
        if lbl_path.exists():
            import shutil
            shutil.copy2(str(img_path), dst_img_dir / img_path.name)
            shutil.copy2(str(lbl_path), dst_label_dir / lbl_path.name)
            count += 1
    return count


def process_ovejas():
    import shutil
    src_img = Path("C:/Users/mabad/Desktop/Ovejas/images")
    src_lbl = Path("C:/Users/mabad/Desktop/Ovejas/annotations")
    dst_img = Path(OUT_DIR) / "images"
    dst_lbl = Path(OUT_DIR) / "labels"

    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    xml_files = list(src_lbl.glob("*.xml"))
    count = 0
    for xml_path in xml_files:
        yolo_lines = convert_xml(xml_path, CLASS_MAP_VOC)
        if yolo_lines:
            img_name = xml_path.stem + ".png"
            img_path = src_img / img_name
            if img_path.exists():
                shutil.copy2(str(img_path), str(dst_img / img_name))
                with open(dst_lbl / (xml_path.stem + ".txt"), "w") as f:
                    f.write("\n".join(yolo_lines))
                count += 1
    print(f"Ovejas: {count} imagenes con anotaciones convertidas a YOLO")


def process_cerdos():
    import shutil
    src_img = Path("C:/Users/mabad/Desktop/Cerdos/pig/train/images")
    src_lbl = Path("C:/Users/mabad/Desktop/Cerdos/pig/train/labels")
    dst_img = Path(OUT_DIR) / "images"
    dst_lbl = Path(OUT_DIR) / "labels"

    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    images = list(src_img.glob("*.jpg")) + list(src_img.glob("*.png")) + list(src_img.glob("*.jpeg"))
    count = 0
    for img_path in images:
        lbl_path = src_lbl / (img_path.stem + ".txt")
        if lbl_path.exists():
            shutil.copy2(str(img_path), str(dst_img / img_path.name))
            shutil.copy2(str(lbl_path), str(dst_lbl / lbl_path.name))
            count += 1
    print(f"Cerdos: {count} imagenes con anotaciones YOLO copiadas")


def process_raw_images():
    import shutil
    for class_name, src_dir in RAW_DIRS.items():
        if class_name in ["ovino", "porcino"]:
            continue

        dst_img = Path(OUT_DIR) / "images"
        dst_lbl = Path(OUT_DIR) / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        images = list(Path(src_dir).glob("*.jpg")) + list(Path(src_dir).glob("*.png"))
        count = 0
        for img in images:
            shutil.copy2(str(img), str(dst_img / img.name))
            lbl_path = dst_lbl / (img.stem + ".txt")
            if not lbl_path.exists():
                lbl_path.touch()
            count += 1
        print(f"{class_name}: {count} imagenes sin anotaciones copiadas")


if __name__ == "__main__":
    print("=== Convirtiendo dataset personalizado ===\n")

    process_ovejas()
    process_cerdos()
    process_raw_images()

    all_images = list(Path(f"{OUT_DIR}/images").glob("*"))
    all_labels = list(Path(f"{OUT_DIR}/labels").glob("*.txt"))
    print(f"\nTotal: {len(all_images)} imagenes, {len(all_labels)} labels")
    print(f"Ubicacion: {OUT_DIR}/")
