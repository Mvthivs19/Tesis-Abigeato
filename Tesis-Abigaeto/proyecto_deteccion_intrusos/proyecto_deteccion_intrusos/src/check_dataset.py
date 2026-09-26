from pathlib import Path

labels_dir = Path("data/custom_yolo/labels")
class_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
names = {0: "humano", 1: "bovino", 2: "equino", 3: "ovino", 4: "porcino"}

for split in ["train", "val", "test"]:
    n_imgs = 0
    for f in (labels_dir / split).glob("*.txt"):
        if f.stat().st_size > 0:
            n_imgs += 1
        for line in f.read_text().splitlines():
            parts = line.strip().split()
            if parts:
                cls = int(parts[0])
                if cls in class_counts:
                    class_counts[cls] += 1
    print(f"  {split}: {n_imgs} imagenes con anotaciones")

print()
print("Distribucion de clases:")
total = 0
for cls_id, count in class_counts.items():
    print(f"  {names[cls_id]}: {count}")
    total += count
print(f"  Total: {total}")
