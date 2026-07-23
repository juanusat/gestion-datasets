#!/usr/bin/env python3
import os
import random
import argparse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

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

def get_expected_image_path(label_path):
    for ext in IMG_EXTENSIONS:
        img_same_dir = label_path.with_suffix(ext)
        if img_same_dir.exists():
            return img_same_dir
            
    parts = list(label_path.parts)
    try:
        idx = len(parts) - 1 - parts[::-1].index('labels')
        parts[idx] = 'images'
        for ext in IMG_EXTENSIONS:
            img_sep = Path(*parts).with_suffix(ext)
            if img_sep.exists():
                return img_sep
    except ValueError:
        pass

    return None

def load_dataset_class_names(dataset_path):
    names_map = {}
    yaml_file = dataset_path / "data.yaml"
    classes_file = dataset_path / "classes.txt"
    
    if yaml_file.exists():
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                in_names = False
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("names:"):
                        in_names = True
                        if "[" in stripped and "]" in stripped:
                            raw = stripped.split("[")[1].split("]")[0]
                            items = [i.strip().strip("'\"") for i in raw.split(",")]
                            for idx, name in enumerate(items):
                                names_map[idx] = name
                            break
                        continue
                    if in_names:
                        if stripped and not line.startswith(" ") and not line.startswith("\t"):
                            break
                        if ":" in stripped:
                            k, v = stripped.split(":", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            if k.isdigit():
                                names_map[int(k)] = v
        except Exception:
            pass
            
    elif classes_file.exists():
        try:
            with open(classes_file, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    name = line.strip()
                    if name:
                        names_map[idx] = name
        except Exception:
            pass

    return names_map

def sample_txt_files(base_dir, n):
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Error: La ruta '{base_dir}' no existe.")
        return

    print(f"Explorando el directorio '{base_dir}'...")
    txt_by_dir = defaultdict(list)
    
    for txt_file in base_path.rglob("*.txt"):
        if is_label_file(txt_file):
            txt_by_dir[txt_file.parent].append(txt_file)

    if not txt_by_dir:
        print("No se encontraron archivos .txt de etiquetas.")
        return

    for directory, files in txt_by_dir.items():
        print(f"\n{'='*70}")
        print(f"Directorio: {directory}")
        print(f"Total de archivos .txt encontrados: {len(files)}")
        print(f"{'='*70}")

        sampled = random.sample(files, min(n, len(files)))
        for f in sampled:
            print(f"\n---> Archivo: {f.name}")
            print(f"---> Ruta: {f}")
            print("-" * 30)
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    content = file.read()
                    print(content.strip() if content.strip() else "<El archivo está vacío>")
            except Exception as e:
                print(f"<Error al leer el archivo: {e}>")
            print("-" * 30)

def _process_file_batch(file_batch):
    local_counts = defaultdict(lambda: defaultdict(int))
    for txt_path_str, dataset_name in file_batch:
        try:
            with open(txt_path_str, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts and parts[0].isdigit():
                        local_counts[dataset_name][int(parts[0])] += 1
        except Exception:
            pass
    return local_counts

def list_classes_per_dataset(base_dir):
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Error: La ruta '{base_dir}' no existe.")
        return

    print("Buscando etiquetas y mapeando datasets...")
    tasks = []
    
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".txt"):
                txt_path = Path(root) / file
                if not is_label_file(txt_path):
                    continue
                try:
                    rel_path = txt_path.relative_to(base_path)
                    dataset_name = rel_path.parts[0]
                except ValueError:
                    dataset_name = "unknown_dataset"

                tasks.append((str(txt_path), dataset_name))

    if not tasks:
        print("No se encontraron archivos .txt de etiquetas.")
        return

    print(f"Procesando {len(tasks)} archivos en paralelo...")
    
    batch_size = max(1, len(tasks) // (os.cpu_count() or 4))
    batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]

    dataset_counts = defaultdict(lambda: defaultdict(int))

    with ProcessPoolExecutor() as executor:
        results = executor.map(_process_file_batch, batches)
        for res in results:
            for ds, counts in res.items():
                for cls_id, count in counts.items():
                    dataset_counts[ds][cls_id] += count

    print("\n=== CONTEO DE CLASES POR DATASET ===")
    for dataset, counts in sorted(dataset_counts.items()):
        ds_path = base_path / dataset
        class_names = load_dataset_class_names(ds_path)
        
        print(f"\nDataset: {dataset}")
        print("-" * 40)
        total_boxes = 0
        for cls_id, count in sorted(counts.items()):
            c_name = class_names.get(cls_id, "Desconocida")
            print(f"  Clase {cls_id} ({c_name}): {count} apariciones")
            total_boxes += count
        print(f"  Total anotaciones: {total_boxes}")

def list_images_missing_labels(base_dir):
    base_path = Path(base_dir)
    missing_count = 0

    print("\n=== IMÁGENES SIN ARCHIVO DE ETIQUETA ===")
    for img_file in base_path.rglob("*"):
        if img_file.suffix.lower() not in IMG_EXTENSIONS:
            continue
        
        label_file = get_expected_label_path(img_file)
        if not label_file.exists():
            print(img_file)
            missing_count += 1
            
    if missing_count == 0:
        print("¡Todo perfecto! Todas las imágenes tienen su archivo .txt correspondiente.")
    else:
        print(f"\nTotal: {missing_count} imágenes sin etiquetas.")

def list_images_with_empty_labels(base_dir):
    base_path = Path(base_dir)
    empty_count = 0

    print("\n=== IMÁGENES CON ETIQUETAS VACÍAS O INVÁLIDAS ===")
    for img_file in base_path.rglob("*"):
        if img_file.suffix.lower() not in IMG_EXTENSIONS:
            continue
        
        label_file = get_expected_label_path(img_file)
        if label_file.exists():
            is_valid = False
            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts and parts[0].lstrip('-').isdigit():
                            is_valid = True
                            break
            except Exception:
                pass
                
            if not is_valid:
                print(f"Imagen: {img_file} --> Etiqueta vacía: {label_file}")
                empty_count += 1

    if empty_count == 0:
        print("No se encontraron etiquetas vacías para las imágenes existentes.")
    else:
        print(f"\nTotal: {empty_count} imágenes con etiquetas que no contienen clases.")

def export_sampled_boxes(base_dir, n, out_dir="xk-out"):
    if not HAS_PIL:
        print("Error: Se requiere Pillow")
        return

    base_path = Path(base_dir)
    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Indexando anotaciones por clase y por dataset...")
    dataset_class_files = defaultdict(lambda: defaultdict(list))
    
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".txt"):
                txt_path = Path(root) / file
                if not is_label_file(txt_path):
                    continue
                try:
                    rel_path = txt_path.relative_to(base_path)
                    dataset_name = rel_path.parts[0]
                except ValueError:
                    dataset_name = "unknown_dataset"

                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        seen_classes = set()
                        for line in f:
                            parts = line.strip().split()
                            if parts and parts[0].isdigit():
                                seen_classes.add(int(parts[0]))
                        for cls_id in seen_classes:
                            dataset_class_files[dataset_name][cls_id].append(txt_path)
                except Exception:
                    pass

    if not dataset_class_files:
        print("No se encontraron etiquetas válidas.")
        return

    COLORS = [
        "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
        "#FFA500", "#800080", "#008000", "#000080", "#FFC0CB", "#A52A2A"
    ]

    total_images_exported = 0
    total_cuts_exported = 0
    print(f"\nGenerando muestras dibujadas y recortes en '{out_dir}' ({n} por clase/dataset)...")

    for dataset_name, class_dict in dataset_class_files.items():
        ds_path = base_path / dataset_name
        class_names = load_dataset_class_names(ds_path)

        ds_out_dir = output_path / dataset_name
        ds_cuts_dir = ds_out_dir / "cuts"
        ds_out_dir.mkdir(parents=True, exist_ok=True)
        ds_cuts_dir.mkdir(parents=True, exist_ok=True)

        for cls_id, txt_files in class_dict.items():
            cls_name = class_names.get(cls_id, f"clase_{cls_id}")
            sampled = random.sample(txt_files, min(n, len(txt_files)))

            for idx, txt_path in enumerate(sampled, 1):
                img_path = get_expected_image_path(txt_path)
                if not img_path:
                    continue

                try:
                    with Image.open(img_path) as img:
                        img = img.convert("RGB")
                        width, height = img.size

                        drawn_img = img.copy()
                        draw = ImageDraw.Draw(drawn_img)

                        box_idx = 0
                        with open(txt_path, 'r', encoding='utf-8') as f:
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
                                        cropped.save(ds_cuts_dir / cut_name)
                                        total_cuts_exported += 1
                                        box_idx += 1

                        out_name = f"cls{cls_id}_{cls_name}_sample{idx}_{img_path.name}"
                        out_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in out_name)
                        save_path = ds_out_dir / out_name
                        drawn_img.save(save_path)
                        total_images_exported += 1
                        print(f" -> Creado: {dataset_name}/{save_path.name}")

                except Exception as e:
                    print(f"Error procesando {img_path}: {e}")

    print(f"\n¡Listo! Exportadas {total_images_exported} imágenes anotadas y {total_cuts_exported} recortes en '{out_dir}'.")

def export_class_config(base_dir, out_dir="config", file_name="labels_origin.yaml"):
    base_path = Path(base_dir)
    config_path = Path(out_dir)
    config_path.mkdir(parents=True, exist_ok=True)

    if not base_path.exists():
        print(f"Error: La ruta '{base_dir}' no existe.")
        return

    print("Indexando datasets y sus clases...")
    dataset_classes = defaultdict(set)

    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".txt"):
                txt_path = Path(root) / file
                if not is_label_file(txt_path):
                    continue
                try:
                    rel_path = txt_path.relative_to(base_path)
                    dataset_name = rel_path.parts[0]
                except ValueError:
                    dataset_name = "unknown_dataset"

                try:
                    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts and parts[0].isdigit():
                                dataset_classes[dataset_name].add(int(parts[0]))
                except Exception:
                    pass

    if not dataset_classes:
        print("No se encontraron clases en los datasets.")
        return

    out_file = config_path / file_name
    lines = []

    for dataset_name, cls_ids in sorted(dataset_classes.items()):
        ds_path = base_path / dataset_name
        class_names = load_dataset_class_names(ds_path)
        
        lines.append(f"{dataset_name}:\n")
        for cls_id in sorted(cls_ids):
            c_name = class_names.get(cls_id, "")
            lines.append(f'  {cls_id}: "{c_name}"\n')
        lines.append("\n")

    with open(out_file, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"\n¡Configuración exportada correctamente en '{out_file}'!")

def main():
    parser = argparse.ArgumentParser(description="Utilidades para exploración y gestión de datasets de imágenes.")
    
    parser.add_argument("-0", dest="sample_n", type=int, metavar="N", 
                        help="Ver N elementos txt al azar de cada subdirectorio, su ruta y contenido.")
    parser.add_argument("-1", dest="opt_1", action="store_true", 
                        help="Listar por cada dataset las clases existentes y su cantidad (optimizado).")
    parser.add_argument("-2", dest="opt_2", action="store_true", 
                        help="Listar los archivos de imagen que no tienen un archivo de etiqueta.")
    parser.add_argument("-3", dest="opt_3", action="store_true", 
                        help="Listar los archivos de imagen que tienen etiqueta pero no contienen clases.")
    parser.add_argument("-4", dest="sample_boxes_n", type=int, metavar="N",
                        help="Exportar N imágenes al azar con sus cajas dibujadas por cada clase y dataset en 'xk-out/'.")
    parser.add_argument("-5", dest="opt_5", action="store_true",
                        help="Exportar a config/labels_origin.yaml las clases existentes por cada dataset.")
    
    parser.add_argument("--dir", type=str, default="origins", 
                        help="Ruta al directorio de los datasets (por defecto: 'origins').")

    args = parser.parse_args()

    if args.sample_n is not None:
        sample_txt_files(args.dir, args.sample_n)
    elif args.opt_1:
        list_classes_per_dataset(args.dir)
    elif args.opt_2:
        list_images_missing_labels(args.dir)
    elif args.opt_3:
        list_images_with_empty_labels(args.dir)
    elif args.sample_boxes_n is not None:
        export_sampled_boxes(args.dir, args.sample_boxes_n)
    elif args.opt_5:
        export_class_config(args.dir)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()