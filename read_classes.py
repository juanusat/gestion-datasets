#!/usr/bin/env python3
"""
Script para identificar las clases detectadas por cada modelo en un directorio
(por defecto: 'xk-versions/').

Funciona sin dependencias externas como PyTorch o Ultralytics, utilizando
únicamente la biblioteca estándar de Python (zipfile + pickle).
"""

import argparse
import json
import os
import pickle
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional


class _DummyTorchObject:
    """Objeto ficticio para desempaquetar checkpoints de PyTorch sin tener torch instalado."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        elif isinstance(state, tuple):
            self._state_tuple = state

    def __getattr__(self, name):
        return None


class _SafeTorchUnpickler(pickle.Unpickler):
    """Unpickler tolerante que captura objetos y tensores de PyTorch sin requerir la librería."""

    def find_class(self, module: str, name: str):
        # Para tipos básicos de Python o colecciones comunes
        if module in ("builtins", "__builtin__"):
            return getattr(sys.modules["builtins"], name)
        if module == "collections" and name in ("OrderedDict", "defaultdict", "deque"):
            import collections
            return getattr(collections, name)
        # Para cualquier clase de PyTorch o Ultralytics
        return _DummyTorchObject

    def persistent_load(self, pid: Any):
        # Los tensores de PyTorch usan persistent_load para referencias de almacenamiento
        return _DummyTorchObject()


def extract_classes_from_pt(pt_path: Path) -> Optional[Dict[int, str]]:
    """Extrae el mapeo {id: nombre_clase} de un archivo checkpoint .pt (YOLO / PyTorch)."""
    # 1. Intentar método nativo sin dependencias (zipfile + pickle)
    if zipfile.is_zipfile(pt_path):
        try:
            with zipfile.ZipFile(pt_path, "r") as z:
                # Localizar archivo data.pkl dentro del zip
                pkl_names = [name for name in z.namelist() if name.endswith("data.pkl")]
                if pkl_names:
                    with z.open(pkl_names[0], "r") as f:
                        data = _SafeTorchUnpickler(f).load()
                        classes = _extract_names_from_checkpoint_dict(data)
                        if classes is not None:
                            return classes
        except Exception as e:
            # Continuar e intentar otros métodos si falla
            pass

    # 2. Si está disponible torch, intentar como respaldo
    try:
        import torch

        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
        classes = _extract_names_from_checkpoint_dict(ckpt)
        if classes is not None:
            return classes
    except Exception:
        pass

    # 3. Si está disponible ultralytics, intentar como respaldo
    try:
        from ultralytics import YOLO

        model = YOLO(str(pt_path))
        if hasattr(model, "names") and model.names:
            return _normalize_names(model.names)
    except Exception:
        pass

    return None


def _extract_names_from_checkpoint_dict(ckpt: Any) -> Optional[Dict[int, str]]:
    """Busca el atributo o clave 'names' dentro de las estructuras habituales de YOLO/PyTorch."""
    if not isinstance(ckpt, dict) and not hasattr(ckpt, "__dict__"):
        return None

    # Caso 1: Diccionario directo con clave 'names'
    if isinstance(ckpt, dict) and "names" in ckpt:
        names = _normalize_names(ckpt["names"])
        if names:
            return names

    # Caso 2: Atributo 'names' en modelo (ckpt['model'])
    model = ckpt.get("model") if isinstance(ckpt, dict) else getattr(ckpt, "model", None)
    if model is not None:
        names = getattr(model, "names", None)
        if names is None and hasattr(model, "__dict__"):
            names = model.__dict__.get("names")
        if names:
            return _normalize_names(names)

    # Caso 3: Atributo 'names' en EMA (ckpt['ema'])
    ema = ckpt.get("ema") if isinstance(ckpt, dict) else getattr(ckpt, "ema", None)
    if ema is not None:
        names = getattr(ema, "names", None)
        if names is None and hasattr(ema, "__dict__"):
            names = ema.__dict__.get("names")
        if names:
            return _normalize_names(names)

    # Caso 4: En el objeto principal si es un modelo directo
    if hasattr(ckpt, "names") and ckpt.names:
        return _normalize_names(ckpt.names)

    return None


def _normalize_names(names: Any) -> Optional[Dict[int, str]]:
    """Convierte listas, tuplas o diccionarios al formato estandarizado {int: str}."""
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in sorted(names.items(), key=lambda item: int(item[0]))}
    elif isinstance(names, (list, tuple)):
        return {idx: str(name) for idx, name in enumerate(names)}
    return None


def inspect_models_directory(models_dir: Path) -> Dict[str, Optional[Dict[int, str]]]:
    """Recorre un directorio y obtiene las clases de todos los modelos encontrados."""
    results: Dict[str, Optional[Dict[int, str]]] = {}
    if not models_dir.exists():
        print(f"Error: El directorio '{models_dir}' no existe.", file=sys.stderr)
        return results

    pt_files = sorted(models_dir.glob("*.pt"))
    if not pt_files:
        print(f"No se encontraron archivos .pt en '{models_dir}'.", file=sys.stderr)
        return results

    for pt_file in pt_files:
        results[pt_file.name] = extract_classes_from_pt(pt_file)

    return results


def print_results(results: Dict[str, Optional[Dict[int, str]]]):
    """Muestra los resultados de forma legible y estructurada en consola."""
    print("=" * 60)
    print("  DETECCIÓN DE CLASES EN MODELOS (.pt)")
    print("=" * 60)

    for filename, classes in results.items():
        print(f"\n📦 Modelo: {filename}")
        if classes is None:
            print("   ⚠️  No se pudieron extraer las clases del modelo.")
            continue

        print(f"   Total de clases: {len(classes)}")
        print("   Clases detectadas:")
        for class_id, class_name in classes.items():
            print(f"     [{class_id}] {class_name}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Identifica las clases detectadas por cada modelo .pt en un directorio."
    )
    parser.add_argument(
        "dir",
        nargs="?",
        default="xk-versions",
        help="Directorio que contiene los modelos (por defecto: xk-versions)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Salida en formato JSON para integración con otros scripts",
    )

    args = parser.parse_args()
    models_dir = Path(args.dir)

    results = inspect_models_directory(models_dir)

    if args.json:
        # Convertir a serializable json
        json_data = {
            model: {str(k): v for k, v in classes.items()} if classes is not None else None
            for model, classes in results.items()
        }
        print(json.dumps(json_data, indent=2, ensure_ascii=False))
    else:
        print_results(results)


if __name__ == "__main__":
    main()
