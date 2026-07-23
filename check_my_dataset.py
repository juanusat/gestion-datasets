#!/usr/bin/env python3
import os
import random
import argparse
from pathlib import Path
from collections import defaultdict

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

COLORS = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
    "#FFA500", "#800080", "#008000", "#000080", "#FFC0CB", "#A52A2A"
]

def load_classes(classes_path):
    class_names = {}
    if os.path.exists(classes_path):
        try:
            with open(classes_path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    name = line.strip()
                    if name:
                        class_names[idx] = name
        except Exception:
            pass
    return class_names

def check_mixed_dataset(n, mixed_dir="my_dataset/mixed", out_dir="xk-out-mixed"):
    if not HAS_PIL:
        print("Error: Se requiere Pillow. Instálalo con 'pip install Pillow'.")
        return

    mixed_path = Path(mixed_dir)
    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    classes_file = mixed_path / "classes.txt"
    class_names = load_classes(classes_file)

    splits = ["train", "val", "test"]

    for split in splits:
        img_dir = mixed_path / "images" / split
        lbl_dir = mixed_path / "labels" / split

        if not img_dir.exists() or not lbl_dir.exists():
            continue

        print(f"\nIndexando partición '{split}'...")
        class_label_files = defaultdict(list)

        for lbl_file in lbl_dir.glob("*.txt"):
            try:
                with open(lbl_file, 'r', encoding='utf-8') as f:
                    seen_classes = set()
                    for line in f:
                        parts = line.strip().split()
                        if parts and parts[0].isdigit():
                            seen_classes.add(int(parts[0]))
                    for cls_id in seen_classes:
                        class_label_files[cls_id].append(lbl_file)
            except Exception:
                pass

        if not class_label_files:
            print(f"No se encontraron etiquetas en '{split}'.")
            continue

        split_out_dir = output_path / split
        split_cuts_dir = split_out_dir / "cuts"
        split_out_dir.mkdir(parents=True, exist_ok=True)
        split_cuts_dir.mkdir(parents=True, exist_ok=True)

        exported_imgs = 0
        exported_cuts = 0

        for cls_id, lbl_files in class_label_files.items():
            cls_name = class_names.get(cls_id, f"clase_{cls_id}")
            sampled = random.sample(lbl_files, min(n, len(lbl_files)))

            for idx, lbl_path in enumerate(sampled, 1):
                img_path = None
                for ext in IMG_EXTENSIONS:
                    candidate = img_dir / f"{lbl_path.stem}{ext}"
                    if candidate.exists():
                        img_path = candidate
                        break

                if not img_path:
                    continue

                try:
                    with Image.open(img_path) as img:
                        img = img.convert("RGB")
                        width, height = img.size

                        drawn_img = img.copy()
                        draw = ImageDraw.Draw(drawn_img)

                        box_idx = 0
                        with open(lbl_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 5 and parts[0].isdigit():
                                    box_cls = int(parts[0])
                                    box_cls_name = class_names.get(box_cls, f"cls_{box_cls}")
                                    x_center = float(parts[1]) * width
                                    y_center = float(parts[2]) * height
                                    w = float(parts[3]) * width
                                    h = float(parts[4]) * height

                                    x1 = max(0, x_center - (w / 2))
                                    y1 = max(0, y_center - (h / 2))
                                    x2 = min(width, x_center + (w / 2))
                                    y2 = min(height, y_center + (h / 2))

                                    color = COLORS[box_cls % len(COLORS)]
                                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                                    label_text = f"{box_cls}: {box_cls_name}"
                                    draw.text((x1 + 3, max(0, y1 - 12)), label_text, fill=color)

                                    if (x2 - x1) > 1 and (y2 - y1) > 1:
                                        cropped = img.crop((int(x1), int(y1), int(x2), int(y2)))
                                        cut_name = f"cls{box_cls}_{box_cls_name}_{img_path.stem}_box{box_idx}{img_path.suffix}"
                                        cut_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in cut_name)
                                        cropped.save(split_cuts_dir / cut_name)
                                        exported_cuts += 1
                                        box_idx += 1

                        out_name = f"cls{cls_id}_{cls_name}_sample{idx}_{img_path.name}"
                        out_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in out_name)
                        save_path = split_out_dir / out_name
                        drawn_img.save(save_path)
                        exported_imgs += 1
                        print(f" -> Creado: {split}/{save_path.name}")

                except Exception as e:
                    print(f"Error procesando {img_path}: {e}")

        print(f"Partición '{split}' completada: {exported_imgs} imágenes anotadas y {exported_cuts} recortes.")

    print(f"\n¡Proceso finalizado! Revisa las imágenes y recortes en '{out_dir}/'.")

def main():
    parser = argparse.ArgumentParser(description="Muestrear e inspeccionar recortes y anotaciones de my_dataset/mixed.")
    parser.add_argument("-n", "--num-samples", type=int, required=True, metavar="N",
                        help="Número N de imágenes a muestrear por cada clase en cada partición (train/val/test).")
    parser.add_argument("--dir", type=str, default="my_dataset/mixed",
                        help="Ruta al dataset mezclado (por defecto: 'my_dataset/mixed').")
    parser.add_argument("--out", type=str, default="xk-out-mixed",
                        help="Carpeta de salida para imágenes y recortes (por defecto: 'xk-out-mixed').")

    args = parser.parse_args()

    check_mixed_dataset(args.num_samples, mixed_dir=args.dir, out_dir=args.out)

if __name__ == "__main__":
    main()
