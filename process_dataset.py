#!/usr/bin/env python3
import os
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict

IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

def is_label_file(filepath):
    name_lower = filepath.name.lower()
    return not (name_lower.startswith("readme") or 
                "license" in name_lower or 
                name_lower == "classes.txt" or 
                name_lower == "data.yaml")

def get_expected_label_path(img_path):
    label_same_dir = img_path.with_suffix('.txt')
    if label_same_dir.exists():
        return label_same_dir
    
    parts = list(img_path.parts)
    try:
        idx = len(parts) - 1 - parts[::-1].index('images')
        parts[idx] = 'labels'
        label_separated = Path(*parts).with_suffix('.txt')
        return label_separated
    except ValueError:
        return label_same_dir

def simple_yaml_load(filepath):
    if not os.path.exists(filepath):
        return {}
    data = {}
    current_key = None
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped == "list:":
                current_key = "list"
                data["list"] = []
                continue
            if current_key == "list" and stripped.startswith("-"):
                item = stripped.lstrip("-").strip().strip("'\"")
                data["list"].append(item)
                continue
            if ":" in stripped:
                parts = stripped.split(":", 1)
                k = parts[0].strip()
                v = parts[1].strip()
                if not v:
                    current_key = k
                    data[current_key] = {}
                else:
                    v = v.strip("'\"")
                    if current_key and isinstance(data.get(current_key), dict):
                        if k.isdigit():
                            data[current_key][int(k)] = v
                        else:
                            data[current_key][k] = v
                    else:
                        data[k] = v
    return data

def build_unified_dataset(origins_dir="origins", config_dir="config", output_dir="my_dataset/full"):
    base_path = Path(origins_dir)
    out_path = Path(output_dir)
    my_dataset_root = Path("my_dataset")

    if out_path.exists():
        shutil.rmtree(out_path)
    out_images_dir = out_path / "images"
    out_labels_dir = out_path / "labels"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    rename_yaml = simple_yaml_load(os.path.join(config_dir, "labels_to_rename.yaml"))
    preserve_yaml = simple_yaml_load(os.path.join(config_dir, "labels_preserve.yaml"))
    
    preserve_list = preserve_yaml.get("list", [])
    unified_classes = sorted(list(set(preserve_list)))
    class_to_id = {cls_name: idx for idx, cls_name in enumerate(unified_classes)}

    with open(out_path / "classes.txt", "w", encoding="utf-8") as f:
        for c in unified_classes:
            f.write(f"{c}\n")

    yaml_lines = ["names:\n"]
    for idx, c in enumerate(unified_classes):
        yaml_lines.append(f'  {idx}: "{c}"\n')
    with open(out_path / "data.yaml", "w", encoding="utf-8") as f:
        f.writelines(yaml_lines)

    copied_stats = defaultdict(lambda: defaultdict(int))
    total_stats = defaultdict(int)

    for img_file in base_path.rglob("*"):
        if img_file.suffix.lower() not in IMG_EXTENSIONS:
            continue
        
        label_file = get_expected_label_path(img_file)
        if not label_file.exists():
            continue

        try:
            rel_path = img_file.relative_to(base_path)
            dataset_name = rel_path.parts[0]
        except ValueError:
            dataset_name = "unknown_dataset"

        ds_rename_map = rename_yaml.get(dataset_name, {})

        new_label_lines = []
        file_classes = set()

        try:
            with open(label_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts or not parts[0].isdigit():
                        continue
                    
                    orig_id = int(parts[0])
                    renamed_label = ds_rename_map.get(orig_id, "").strip()

                    if renamed_label in class_to_id:
                        new_id = class_to_id[renamed_label]
                        new_line = f"{new_id} " + " ".join(parts[1:]) + "\n"
                        new_label_lines.append(new_line)
                        file_classes.add((renamed_label, new_id))
        except Exception:
            continue

        if not new_label_lines:
            continue

        unique_file_id = f"{dataset_name}_{img_file.stem}"
        new_img_name = f"{unique_file_id}{img_file.suffix}"
        new_lbl_name = f"{unique_file_id}.txt"

        shutil.copy2(img_file, out_images_dir / new_img_name)
        with open(out_labels_dir / new_lbl_name, "w", encoding="utf-8") as f:
            f.writelines(new_label_lines)

        for line in new_label_lines:
            parts = line.strip().split()
            new_id = int(parts[0])
            cls_name = unified_classes[new_id]
            copied_stats[cls_name][dataset_name] += 1
            total_stats[cls_name] += 1

    all_datasets = sorted(list(set(ds for ds_map in copied_stats.values() for ds in ds_map.keys())))
    
    stats_md_path = out_path / "stats.md"
    md_lines = []
    md_lines.append("# Estadísticas del Dataset Unificado (`my_dataset/full`)\n\n")
    
    header = "| Clase | ID Unificado | " + " | ".join(all_datasets) + " | Total |"
    separator = "| --- | --- | " + " | ".join(["---"] * len(all_datasets)) + " | --- |"
    md_lines.append(header + "\n")
    md_lines.append(separator + "\n")

    for cls_name in unified_classes:
        cid = class_to_id[cls_name]
        ds_counts = [str(copied_stats[cls_name][ds]) for ds in all_datasets]
        tot = total_stats[cls_name]
        row = f"| {cls_name} | {cid} | " + " | ".join(ds_counts) + f" | {tot} |"
        md_lines.append(row + "\n")

    total_boxes = sum(total_stats.values())
    tot_row = f"| **TOTAL** | - | " + " | ".join([str(sum(copied_stats[c][ds] for c in unified_classes)) for ds in all_datasets]) + f" | **{total_boxes}** |"
    md_lines.append(tot_row + "\n")

    with open(stats_md_path, "w", encoding="utf-8") as f:
        f.writelines(md_lines)

    print(f"Dataset unificado creado con éxito en '{output_dir}'.")
    print(f"Estadísticas guardadas en '{stats_md_path}'.")

def mix_dataset(ratios, src_dir="my_dataset/full", dest_dir="my_dataset/mixed"):
    if len(ratios) != 3 or sum(ratios) != 100:
        print("Error: El parámetro -c debe recibir 3 valores enteros que sumen 100 (ej: -c 70 20 10).")
        return

    train_pct, val_pct, test_pct = ratios
    src_path = Path(src_dir)
    dest_path = Path(dest_dir)

    if dest_path.exists():
        shutil.rmtree(dest_path)

    for split in ["train", "val", "test"]:
        (dest_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (dest_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    src_img_dir = src_path / "images"
    src_lbl_dir = src_path / "labels"

    if not src_img_dir.exists():
        print(f"Error: La ruta '{src_img_dir}' no existe. Ejecuta primero la unificación.")
        return

    classes_file = src_path / "classes.txt"
    class_names = []
    if classes_file.exists():
        with open(classes_file, "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f if line.strip()]

    class_to_images = defaultdict(list)
    all_images = set()

    for img_p in src_img_dir.glob("*"):
        if img_p.suffix.lower() not in IMG_EXTENSIONS:
            continue
        all_images.add(img_p)
        lbl_p = src_lbl_dir / f"{img_p.stem}.txt"
        if lbl_p.exists():
            try:
                with open(lbl_p, "r", encoding="utf-8") as f:
                    file_cls = set()
                    for line in f:
                        parts = line.strip().split()
                        if parts and parts[0].isdigit():
                            file_cls.add(int(parts[0]))
                    for cid in file_cls:
                        class_to_images[cid].append(img_p)
            except Exception:
                pass

    assigned_split = {}

    for cid in sorted(class_to_images.keys()):
        imgs_of_cls = class_to_images[cid]
        unassigned = [img for img in imgs_of_cls if img not in assigned_split]
        random.shuffle(unassigned)

        total_u = len(unassigned)
        if total_u == 0:
            continue

        n_tr = int(total_u * (train_pct / 100.0))
        n_va = int(total_u * (val_pct / 100.0))

        for img in unassigned[:n_tr]:
            assigned_split[img] = "train"
        for img in unassigned[n_tr:n_tr + n_va]:
            assigned_split[img] = "val"
        for img in unassigned[n_tr + n_va:]:
            assigned_split[img] = "test"

    remaining_unassigned = [img for img in all_images if img not in assigned_split]
    random.shuffle(remaining_unassigned)
    tot_rem = len(remaining_unassigned)
    n_tr = int(tot_rem * (train_pct / 100.0))
    n_va = int(tot_rem * (val_pct / 100.0))

    for img in remaining_unassigned[:n_tr]:
        assigned_split[img] = "train"
    for img in remaining_unassigned[n_tr:n_tr + n_va]:
        assigned_split[img] = "val"
    for img in remaining_unassigned[n_tr + n_va:]:
        assigned_split[img] = "test"

    splits = {
        "train": [],
        "val": [],
        "test": []
    }
    for img_p, split_name in assigned_split.items():
        splits[split_name].append(img_p)

    split_class_counts = {
        "train": defaultdict(int),
        "val": defaultdict(int),
        "test": defaultdict(int)
    }

    for split_name, img_list in splits.items():
        for img_p in img_list:
            shutil.copy2(img_p, dest_path / "images" / split_name / img_p.name)
            lbl_p = src_lbl_dir / f"{img_p.stem}.txt"
            if lbl_p.exists():
                shutil.copy2(lbl_p, dest_path / "labels" / split_name / lbl_p.name)
                try:
                    with open(lbl_p, "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts and parts[0].isdigit():
                                cid = int(parts[0])
                                split_class_counts[split_name][cid] += 1
                except Exception:
                    pass

    data_yaml_src = src_path / "data.yaml"
    if data_yaml_src.exists():
        shutil.copy2(data_yaml_src, dest_path / "data.yaml")

    classes_file_src = src_path / "classes.txt"
    if classes_file_src.exists():
        shutil.copy2(classes_file_src, dest_path / "classes.txt")

    total_imgs = len(all_images)
    stats_md_path = dest_path / "stats.md"
    lines = []
    lines.append(f"# Métricas de Partición Estratificada (`-c {' '.join(map(str, ratios))}`)\n\n")
    lines.append("## Resumen de Imágenes\n\n")
    lines.append(f"- **Total de imágenes**: {total_imgs}\n")
    lines.append(f"- **Train**: {len(splits['train'])} imágenes ({len(splits['train'])/max(1,total_imgs)*100:.1f}%)\n")
    lines.append(f"- **Val**: {len(splits['val'])} imágenes ({len(splits['val'])/max(1,total_imgs)*100:.1f}%)\n")
    lines.append(f"- **Test**: {len(splits['test'])} imágenes ({len(splits['test'])/max(1,total_imgs)*100:.1f}%)\n\n")

    lines.append("## Conteo de Anotaciones por Clase\n\n")
    lines.append("| ID Clase | Nombre de Clase | Train | Val | Test | Total |\n")
    lines.append("| --- | --- | --- | --- | --- | --- |\n")

    all_cls_ids = sorted(list(set(cid for s in split_class_counts.values() for cid in s.keys())))
    if class_names:
        for idx in range(len(class_names)):
            if idx not in all_cls_ids:
                all_cls_ids.append(idx)
        all_cls_ids = sorted(list(set(all_cls_ids)))

    tot_train = 0
    tot_val = 0
    tot_test = 0

    for cid in all_cls_ids:
        cname = class_names[cid] if cid < len(class_names) else f"Clase_{cid}"
        tr = split_class_counts["train"][cid]
        va = split_class_counts["val"][cid]
        te = split_class_counts["test"][cid]
        tot = tr + va + te

        tot_train += tr
        tot_val += va
        tot_test += te

        lines.append(f"| {cid} | {cname} | {tr} | {va} | {te} | {tot} |\n")

    lines.append(f"| **TOTAL** | - | **{tot_train}** | **{tot_val}** | **{tot_test}** | **{tot_train + tot_val + tot_test}** |\n")

    with open(stats_md_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Mezcla y partición completada con éxito en '{dest_dir}'.")
    print(f"Estadísticas guardadas en '{stats_md_path}'.")

def main():
    parser = argparse.ArgumentParser(description="Script para unificar datasets y generar particiones train/val/test.")
    parser.add_argument("-u", "--unify", action="store_true", help="Unificar datasets de origins/ a my_dataset/full")
    parser.add_argument("-c", "--split", nargs=3, type=int, metavar=('TRAIN', 'VAL', 'TEST'),
                        help="Mezclar y dividir my_dataset/full hacia my_dataset/mixed (ej: -c 70 20 10)")
    
    args = parser.parse_args()

    if args.unify:
        build_unified_dataset()
    elif args.split:
        mix_dataset(args.split)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
