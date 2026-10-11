#!/usr/bin/env python3
import os
import sys
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any

IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}


def is_label_file(filepath: Path) -> bool:
    """Determina si un archivo .txt es un archivo de anotaciones valido."""
    name_lower = filepath.name.lower()
    return not (
        name_lower.startswith("readme")
        or "license" in name_lower
        or name_lower == "classes.txt"
        or name_lower == "data.yaml"
    )


def get_expected_label_path(img_path: Path) -> Path:
    """Encuentra la ruta esperada del archivo de etiquetas para una imagen."""
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


def simple_yaml_load(filepath: str) -> Dict[str, Any]:
    """Cargador ligero de archivos YAML compatible con la estructura de configuracion del proyecto."""
    if not os.path.exists(filepath):
        return {}
    data: Dict[str, Any] = {}
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


def resolve_paths(fotos_dir_arg: Optional[str], origins_dir_arg: Optional[str], config_dir_arg: Optional[str]) -> Tuple[Path, Path, Path]:
    """Resuelve rutas relativas y absolutas para fotos, origins y config."""
    script_dir = Path(__file__).resolve().parent

    # 1. Fotos dir
    if fotos_dir_arg:
        fotos_path = Path(fotos_dir_arg).resolve()
    else:
        candidates = [
            script_dir.parent / "etiquetado" / "fotos",
            script_dir / "fotos",
            Path("/home/juan/h/pdu/etiquetado/fotos"),
        ]
        fotos_path = next((p for p in candidates if p.exists()), candidates[0])

    # 2. Origins dir
    if origins_dir_arg:
        origins_path = Path(origins_dir_arg).resolve()
    else:
        candidates = [
            script_dir / "origins",
            script_dir.parent / "gestion-datasets" / "origins",
            Path("/home/juan/h/pdu/gestion-datasets/origins"),
        ]
        origins_path = next((p for p in candidates if p.exists()), candidates[0])

    # 3. Config dir
    if config_dir_arg:
        config_path = Path(config_dir_arg).resolve()
    else:
        candidates = [
            script_dir / "config",
            script_dir.parent / "gestion-datasets" / "config",
            Path("/home/juan/h/pdu/gestion-datasets/config"),
        ]
        config_path = next((p for p in candidates if p.exists()), candidates[0])

    return fotos_path, origins_path, config_path


def load_classes(config_dir: Path, fotos_dir: Path) -> List[str]:
    """Carga las clases preservadas unificadas desde config o clases.txt."""
    preserve_yaml = simple_yaml_load(str(config_dir / "labels_preserve.yaml"))
    preserve_list = preserve_yaml.get("list", [])
    if preserve_list:
        return sorted(list(set(preserve_list)))

    classes_txt = fotos_dir.parent / "clases.txt"
    if classes_txt.exists():
        with open(classes_txt, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            if lines:
                return lines

    return ["sin_botas", "sin_casco", "sin_chaleco", "sin_gafas", "sin_guantes"]


def analyze_fotos_directory(fotos_dir: Path, classes: List[str]) -> Dict[str, Any]:
    """
    Analiza el directorio fotos para obtener metricas completas:
    - Total de imagenes
    - Conteo de instancias por clase
    - Conteo de imagenes por clase
    - Deteccion de imagenes de apoyo (__plus)
    """
    if not fotos_dir.exists():
        return {
            "exists": False,
            "total_images": 0,
            "class_counts": {c: 0 for c in classes},
            "class_image_counts": {c: 0 for c in classes},
            "plus_images": [],
            "plus_labels": [],
            "all_images": [],
        }

    class_to_id = {c: i for i, c in enumerate(classes)}
    class_counts = defaultdict(int)
    class_image_counts = defaultdict(int)
    plus_images = []
    plus_labels = []
    all_images = []

    for img_p in sorted(fotos_dir.rglob("*")):
        if img_p.suffix.lower() not in IMG_EXTENSIONS:
            continue

        lbl_p = img_p.with_suffix('.txt')
        if not lbl_p.exists():
            continue

        all_images.append(img_p)
        is_plus = "__plus" in img_p.stem
        if is_plus:
            plus_images.append(img_p)
            if lbl_p.exists() and lbl_p not in plus_labels:
                plus_labels.append(lbl_p)

        file_classes: Set[int] = set()
        try:
            with open(lbl_p, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts and parts[0].isdigit():
                        cid = int(parts[0])
                        if 0 <= cid < len(classes):
                            cname = classes[cid]
                            class_counts[cname] += 1
                            file_classes.add(cid)
        except Exception:
            pass

        for cid in file_classes:
            class_image_counts[classes[cid]] += 1

    # Asegurar todas las clases presentes
    for c in classes:
        if c not in class_counts:
            class_counts[c] = 0
        if c not in class_image_counts:
            class_image_counts[c] = 0

    return {
        "exists": True,
        "total_images": len(all_images),
        "class_counts": dict(class_counts),
        "class_image_counts": dict(class_image_counts),
        "plus_images": plus_images,
        "plus_labels": plus_labels,
        "all_images": all_images,
    }


def print_fotos_count_report(fotos_stats: Dict[str, Any], classes: List[str]):
    """Muestra el reporte detallado del conteo actual de fotos/."""
    print(f"\n{'='*70}")
    print(f"📊 CONTEO DE IMÁGENES E INSTANCIAS EN FOTOS")
    print(f"{'='*70}")
    print(f"Total imágenes válidas: {fotos_stats['total_images']}")
    print(f"Imágenes de apoyo (__plus): {len(fotos_stats['plus_images'])}")
    print()
    header = f"{'ID':<4} | {'Clase':<16} | {'Instancias':<12} | {'Imágenes':<10}"
    divider = f"{'-'*4}-+-{'-'*16}-+-{'-'*12}-+-{'-'*10}"
    print(header)
    print(divider)
    tot_inst = sum(fotos_stats['class_counts'].values())
    for idx, c in enumerate(classes):
        cnt = fotos_stats['class_counts'].get(c, 0)
        imgs = fotos_stats['class_image_counts'].get(c, 0)
        print(f"{idx:<4} | {c:<16} | {cnt:<12} | {imgs:<10}")
    print(divider)
    print(f"{'':<4}   {'TOTAL':<16} | {tot_inst:<12} | {fotos_stats['total_images']:<10}")
    print(f"{'='*70}\n")


def clean_support_instances(fotos_dir: Path, classes: List[str], silent: bool = False) -> bool:
    """Elimina las imagenes y etiquetas de apoyo que contengan el sufijo '__plus'."""
    if not silent:
        print(f"\n{'='*70}")
        print(f"LIMPIEZA DE IMÁGENES DE APOYO (__plus)")
        print(f"{'='*70}")
        print(f"Directorio objetivo: {fotos_dir}")

    if not fotos_dir.exists():
        if not silent:
            print(f"Error: El directorio '{fotos_dir}' no existe.")
        return False

    plus_img_files = []
    plus_lbl_files = []

    for f in fotos_dir.rglob("*"):
        if "__plus" in f.stem:
            if f.suffix.lower() in IMG_EXTENSIONS:
                plus_img_files.append(f)
            elif f.suffix.lower() == ".txt":
                plus_lbl_files.append(f)

    if not plus_img_files and not plus_lbl_files:
        if not silent:
            print("✓ No se encontraron imágenes ni etiquetas con el sufijo '__plus' en el directorio.")
        return True

    if not silent:
        print(f"Archivos de apoyo encontrados:")
        print(f"  - Imágenes (__plus): {len(plus_img_files)}")
        print(f"  - Etiquetas (.txt): {len(plus_lbl_files)}")

    deleted_imgs = 0
    deleted_lbls = 0

    for img_p in plus_img_files:
        try:
            img_p.unlink()
            deleted_imgs += 1
        except Exception as e:
            if not silent:
                print(f"  [ERROR] No se pudo eliminar imagen {img_p.name}: {e}")

    for lbl_p in plus_lbl_files:
        try:
            lbl_p.unlink()
            deleted_lbls += 1
        except Exception as e:
            if not silent:
                print(f"  [ERROR] No se pudo eliminar etiqueta {lbl_p.name}: {e}")

    if not silent:
        print(f"\n✓ Se eliminaron {deleted_imgs} imágenes y {deleted_lbls} archivos de anotación exitosamente.")
        after_stats = analyze_fotos_directory(fotos_dir, classes)
        print_fotos_count_report(after_stats, classes)

    return True


def index_candidate_instances(
    origins_dir: Path,
    config_dir: Path,
    classes: List[str],
    existing_stems: Set[str]
) -> List[Dict[str, Any]]:
    """
    Escanea origins/ y convierte anotaciones segun labels_to_rename.yaml y labels_preserve.yaml.
    Retorna la lista de candidatos validos con clases preservadas.
    """
    rename_yaml = simple_yaml_load(str(config_dir / "labels_to_rename.yaml"))
    class_to_id = {c: i for i, c in enumerate(classes)}

    candidates = []

    for img_file in sorted(origins_dir.rglob("*")):
        if img_file.suffix.lower() not in IMG_EXTENSIONS:
            continue

        lbl_file = get_expected_label_path(img_file)
        if not lbl_file.exists():
            continue

        try:
            rel_path = img_file.relative_to(origins_dir)
            dataset_name = rel_path.parts[0]
        except ValueError:
            dataset_name = "unknown_dataset"

        ds_rename_map = rename_yaml.get(dataset_name, {})
        new_label_lines = []
        file_classes = defaultdict(int)

        try:
            with open(lbl_file, 'r', encoding='utf-8', errors='ignore') as f:
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
                        file_classes[new_id] += 1
        except Exception:
            continue

        if not new_label_lines:
            continue

        unique_stem = f"{dataset_name}_{img_file.stem}__plus"
        if unique_stem in existing_stems:
            continue

        candidates.append({
            "img_path": img_file,
            "dataset_name": dataset_name,
            "target_stem": unique_stem,
            "target_img_name": f"{unique_stem}{img_file.suffix}",
            "target_lbl_name": f"{unique_stem}.txt",
            "converted_lines": new_label_lines,
            "class_counts": dict(file_classes),
            "primary_classes": set(file_classes.keys()),
        })

    return candidates


def balance_and_select_candidates(
    candidates: List[Dict[str, Any]],
    current_counts: Dict[str, int],
    classes: List[str],
    threshold: int = 25,
    compensate_mode: Optional[int] = 0,
) -> Tuple[List[Dict[str, Any]], Dict[int, int], Dict[int, int], Dict[int, int], Set[int]]:
    """
    Calcula objetivos de balanceo con tolerancia +-threshold e identifica de forma óptima
    el conjunto de candidatos a incorporar sin sobrepasar clases que ya alcanzaron su meta.
    - Si compensate_mode == 0: la meta base es la clase con más instancias de forma natural.
    - Si compensate_mode > 0: la meta base es ese valor específico (ej. 1200).
    """
    class_to_id = {c: i for i, c in enumerate(classes)}
    current_by_id = {class_to_id[c]: current_counts.get(c, 0) for c in classes}

    if compensate_mode is None or compensate_mode == 0:
        base_target = max(current_by_id.values())
    else:
        base_target = compensate_mode

    # Generar objetivo por clase con umbral aleatorio [-threshold, +threshold]
    # y fijar techo estricto (max_allowed) para no inflar clases ya satisfechas
    targets: Dict[int, int] = {}
    max_allowed: Dict[int, int] = {}
    for cid in range(len(classes)):
        offset = random.randint(-threshold, threshold)
        if current_by_id[cid] > base_target:
            # La clase ya supera la meta base deseada: se mantiene y no debe exceder su inicial + threshold
            targets[cid] = current_by_id[cid]
            max_allowed[cid] = current_by_id[cid] + threshold
        else:
            targets[cid] = max(current_by_id[cid], base_target + offset)
            max_allowed[cid] = targets[cid] + threshold

    # Selección voraz optimizada con restricción estricta de sobrepaso
    sim_counts = dict(current_by_id)
    added_counts = defaultdict(int)
    selected_candidates: List[Dict[str, Any]] = []
    available = list(candidates)
    random.shuffle(available)
    exhausted_classes: Set[int] = set()

    while True:
        deficits = {
            cid: targets[cid] - sim_counts[cid]
            for cid in range(len(classes))
            if cid not in exhausted_classes
        }
        active_deficits = {cid: d for cid, d in deficits.items() if d > 0}

        if not active_deficits:
            break

        sorted_deficits = sorted(active_deficits.keys(), key=lambda c: active_deficits[c], reverse=True)

        picked = None
        for cid in sorted_deficits:
            matching = [c for c in available if cid in c["class_counts"]]
            if not matching:
                exhausted_classes.add(cid)
                continue

            well_suited = []
            for cand in matching:
                overshoot = False
                for other_cid, count in cand["class_counts"].items():
                    if (sim_counts[other_cid] + count) > max_allowed[other_cid]:
                        overshoot = True
                        break
                if not overshoot:
                    well_suited.append(cand)

            if not well_suited:
                # No hay candidatos que puedan aportar a cid sin sobrepasar alguna otra clase
                exhausted_classes.add(cid)
                continue

            # Priorizar candidatos que aporten menor penalización en clases sin déficit
            def candidate_cost(cand):
                cost = 0
                for c_id, cnt in cand["class_counts"].items():
                    if c_id not in active_deficits:
                        cost += cnt * 10
                return cost

            well_suited.sort(key=candidate_cost)
            min_cost = candidate_cost(well_suited[0])
            best_pool = [c for c in well_suited if candidate_cost(c) == min_cost]
            picked = random.choice(best_pool)
            break

        if picked is None:
            break

        selected_candidates.append(picked)
        available.remove(picked)

        for cid, count in picked["class_counts"].items():
            sim_counts[cid] += count
            added_counts[cid] += count

    return selected_candidates, targets, dict(added_counts), sim_counts, exhausted_classes


def apply_candidate_transfer(
    selected_candidates: List[Dict[str, Any]],
    fotos_dir: Path,
    dry_run: bool = False
) -> int:
    """Copia las imagenes seleccionadas y escribe los archivos de etiquetas YOLO en fotos/."""
    if dry_run:
        return len(selected_candidates)

    fotos_dir.mkdir(parents=True, exist_ok=True)
    copied_count = 0

    for cand in selected_candidates:
        src_img: Path = cand["img_path"]
        dst_img = fotos_dir / cand["target_img_name"]
        dst_lbl = fotos_dir / cand["target_lbl_name"]

        shutil.copy2(src_img, dst_img)

        with open(dst_lbl, "w", encoding="utf-8") as f:
            f.writelines(cand["converted_lines"])

        copied_count += 1

    return copied_count


def print_summary_report(
    classes: List[str],
    initial_counts: Dict[str, int],
    targets: Dict[int, int],
    added_counts: Dict[int, int],
    final_counts: Dict[int, int],
    selected_candidates: List[Dict[str, Any]],
    threshold: int,
    compensate_mode: Optional[int],
    exhausted_classes: Optional[Set[int]] = None,
    dry_run: bool = False
):
    """Muestra el reporte estructurado en consola."""
    class_to_id = {c: i for i, c in enumerate(classes)}

    added_imgs_per_class = defaultdict(int)
    for cand in selected_candidates:
        for cid in cand["class_counts"]:
            added_imgs_per_class[cid] += 1

    mode_str = "[MODO SIMULACIÓN --dry-run]" if dry_run else "[EJECUCIÓN COMPLETADA]"
    target_info = "Equiparar a clase máxima" if (compensate_mode is None or compensate_mode == 0) else f"Meta fija {compensate_mode}"

    print(f"\n{'='*100}")
    print(f"📊 REPORTE DE BALANCEO E INCORPORACIÓN DE INSTANCIAS (__plus) {mode_str}")
    print(f"{'='*100}")
    print(f"• Modo de compensación: {target_info}")
    print(f"• Total de nuevas imágenes incorporadas a fotos/: {len(selected_candidates)}")
    print(f"• Umbral aleatorio de tolerancia: ±{threshold} instancias")
    print()

    header = f"{'ID':<4} | {'Clase':<16} | {'Inicial':<10} | {f'Meta (±{threshold})':<14} | {'Inst. Agregadas':<16} | {'Img. Agregadas':<15} | {'Final':<10} | {'Estado':<15}"
    divider = f"{'-'*4}-+-{'-'*16}-+-{'-'*10}-+-{'-'*14}-+-{'-'*16}-+-{'-'*15}-+-{'-'*10}-+-{'-'*15}"
    print(header)
    print(divider)

    total_init = sum(initial_counts.values())
    total_added_inst = sum(added_counts.values())
    total_final = sum(final_counts.values())

    for idx, cname in enumerate(classes):
        cid = class_to_id[cname]
        init_c = initial_counts.get(cname, 0)
        tgt_c = targets.get(cid, 0)
        add_c = added_counts.get(cid, 0)
        add_img = added_imgs_per_class.get(cid, 0)
        fin_c = final_counts.get(cid, init_c)

        diff = fin_c - tgt_c
        if abs(diff) <= threshold:
            status = "✓ Equilibrada"
        elif diff > threshold:
            status = f"+{diff} Excedida"
        else:
            if exhausted_classes and cid in exhausted_classes:
                status = f"{diff} Agotada"
            else:
                status = f"{diff} Déficit"

        print(f"{idx:<4} | {cname:<16} | {init_c:<10} | {tgt_c:<14} | {add_c:<16} | {add_img:<15} | {fin_c:<10} | {status:<15}")

    print(divider)
    print(f"{'':<4}   {'TOTAL':<16} | {total_init:<10} | {'-':<14} | {total_added_inst:<16} | {len(selected_candidates):<15} | {total_final:<10} |")
    print(f"{'='*100}\n")

    if exhausted_classes:
        print("⚠️  AVISO DE DISPONIBILIDAD DE CANDIDATOS:")
        for cid in sorted(exhausted_classes):
            cname = classes[cid]
            d = targets[cid] - final_counts.get(cid, initial_counts.get(cname, 0))
            print(f"  • {cname}: Se agotaron los candidatos viables en 'origins/' (déficit de {d} instancias respecto a meta).")
            print(f"    No se agregaron más imágenes porque sobrepasarían las clases ya satisfechas.")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Script para equilibrar y agregar más instancias de apoyo (__plus) al directorio 'fotos/' desde 'origins/'."
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="Mostrar únicamente el conteo actual de imágenes e instancias en fotos/ sin modificar archivos."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Borrar todas las imágenes y etiquetas de apoyo que contengan el sufijo '__plus' en 'fotos/'."
    )
    parser.add_argument(
        "-c", "--compensate",
        type=int,
        nargs="?",
        const=0,
        default=None,
        metavar="CANTIDAD",
        help="Compensar instancias: '-c 0' para equilibrar con la clase con más instancias, o '-c NUM' (ej: -c 1200) para buscar llegar a esa cantidad por clase."
    )
    parser.add_argument(
        "-f", "--fotos-dir",
        type=str,
        default=None,
        help="Ruta al directorio 'fotos' donde process_dataset.py lee las imágenes (por defecto: ../etiquetado/fotos)."
    )
    parser.add_argument(
        "-o", "--origins-dir",
        type=str,
        default=None,
        help="Ruta al directorio 'origins' con los datasets de origen (por defecto: origins)."
    )
    parser.add_argument(
        "-k", "--config-dir",
        type=str,
        default=None,
        help="Ruta al directorio 'config' con los archivos YAML de mapeo (por defecto: config)."
    )
    parser.add_argument(
        "-t", "--threshold",
        type=int,
        default=25,
        help="Umbral de tolerancia aleatoria entre clases en número de imágenes/instancias (por defecto: 25)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simular la selección y balanceo sin copiar ni escribir archivos en el disco."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Limpia automáticamente cualquier imagen previa con '__plus' y vuelve a ejecutar el incremento."
    )

    args = parser.parse_args()

    fotos_path, origins_path, config_path = resolve_paths(args.fotos_dir, args.origins_dir, args.config_dir)
    classes = load_classes(config_path, fotos_path)

    # 1. Caso de solo conteo (--count)
    if args.count:
        print(f"\n🔍 Inspeccionando directorio de fotos: {fotos_path}")
        fotos_stats = analyze_fotos_directory(fotos_path, classes)
        if not fotos_stats["exists"]:
            print(f"Error: El directorio '{fotos_path}' no existe o no es accesible.")
            sys.exit(1)
        print_fotos_count_report(fotos_stats, classes)
        sys.exit(0)

    # 2. Caso de limpieza (--clean)
    if args.clean:
        success = clean_support_instances(fotos_path, classes)
        sys.exit(0 if success else 1)

    # 3. Si se pasa --force: limpiar automáticamente y reejecutar
    if args.force:
        print("\n⚙️  Bandera --force detectada: Limpiando instancias de apoyo '__plus' previas antes de reejecutar...")
        clean_support_instances(fotos_path, classes, silent=False)

    # 4. Análisis del estado actual en fotos/
    print(f"\n🔍 Inspeccionando directorio de fotos: {fotos_path}")
    fotos_stats = analyze_fotos_directory(fotos_path, classes)

    if not fotos_stats["exists"]:
        print(f"Error: El directorio '{fotos_path}' no existe o no es accesible.")
        sys.exit(1)

    print(f"✓ Se encontraron {fotos_stats['total_images']} imágenes con anotaciones válidas.")

    # 5. Detección de imágenes previas con __plus
    existing_plus = len(fotos_stats["plus_images"])
    if existing_plus > 0 and not args.force:
        print(f"\n{'!'*70}")
        print(f"⚠️  ATENCIÓN: Se detectaron {existing_plus} imágenes de apoyo con el sufijo '__plus' en 'fotos/'.")
        print(f"Para evitar duplicados o sobre-inyección, la operación se ha detenido.")
        print(f"")
        print(f"Acciones disponibles:")
        print(f"  • Si deseas LIMPIAR las imágenes de apoyo previamente enviadas, ejecuta:")
        print(f"      python add_more_instances.py --clean")
        print(f"  • Si deseas limpiar y RE-EJECUTAR automáticamente el incremento, agrega --force:")
        print(f"      python add_more_instances.py --force")
        print(f"{'!'*70}\n")
        sys.exit(0)

    # Determinar modo de compensación:
    compensate_val = args.compensate if args.compensate is not None else 0

    # 6. Indexar candidatos de origins/
    print(f"📦 Indexando candidatos desde: {origins_path}")
    existing_stems = {img.stem for img in fotos_stats["all_images"]}
    candidates = index_candidate_instances(origins_path, config_path, classes, existing_stems)

    if not candidates:
        print("Error: No se encontraron candidatos con anotaciones válidas en 'origins/'.")
        sys.exit(1)

    print(f"✓ Se indexaron {len(candidates)} candidatos disponibles con anotaciones preservadas.")

    # 7. Ejecutar algoritmo de balanceo y selección
    selected, targets, added_counts, final_sim_counts, exhausted_classes = balance_and_select_candidates(
        candidates=candidates,
        current_counts=fotos_stats["class_counts"],
        classes=classes,
        threshold=args.threshold,
        compensate_mode=compensate_val,
    )

    if not selected:
        if exhausted_classes:
            print("\n⚠️ No se seleccionaron imágenes de apoyo:")
            print("Los candidatos disponibles en 'origins/' sobrepasarían las clases que ya alcanzaron su meta.")
            for cid in sorted(exhausted_classes):
                print(f"  • {classes[cid]}: agotada sin candidatos viables respetando el umbral ±{args.threshold}.")
        else:
            print("\n✓ Las clases ya se encuentran equilibradas dentro del umbral. No se requieren imágenes adicionales.")
        sys.exit(0)

    # 8. Transferir / escribir archivos
    copied_count = apply_candidate_transfer(selected, fotos_path, dry_run=args.dry_run)

    # 9. Imprimir reporte detallado
    print_summary_report(
        classes=classes,
        initial_counts=fotos_stats["class_counts"],
        targets=targets,
        added_counts=added_counts,
        final_counts=final_sim_counts,
        selected_candidates=selected,
        threshold=args.threshold,
        compensate_mode=compensate_val,
        exhausted_classes=exhausted_classes,
        dry_run=args.dry_run
    )

    if not args.dry_run:
        print(f"✓ {copied_count} imágenes y etiquetas con el sufijo '__plus' fueron incorporadas exitosamente a:")
        print(f"  {fotos_path}\n")
        print(f"💡 Puedes ahora volver a procesar tu dataset ejecutando en 'etiquetado/':")
        print(f"  python process_dataset.py -u -c 70 20 10 -e\n")


if __name__ == "__main__":
    main()
