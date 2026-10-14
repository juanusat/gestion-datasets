#!/usr/bin/env python3
import os
import shutil
import random
from pathlib import Path

def create_nuclear_leakage(mixed_dir="my_dataset/mixed", seed=42):
    """
    Induce un Data Leakage del 100% copiando imágenes de 'train' directamente 
    hacia 'val' y 'test'. Mantiene las cuotas numéricas intactas.
    """
    base_path = Path(mixed_dir)
    if not base_path.exists():
        print(f"Error: No se encontró {mixed_dir}")
        return

    print("☢️ Iniciando Clonación Explícita (100% Data Leakage)...")
    
    train_img_dir = base_path / "train" / "images"
    train_lbl_dir = base_path / "train" / "labels"
    
    if not train_img_dir.exists():
        print("Error: La carpeta train/images no existe.")
        return

    # 1. Contar cuotas actuales para no alterar los números del reporte
    quotas = {}
    for split in ["train", "val", "test"]:
        img_dir = base_path / split / "images"
        quotas[split] = len(list(img_dir.glob("*.*"))) if img_dir.exists() else 0

    print(f"📊 Cuotas a simular -> Train: {quotas['train']}, Val: {quotas['val']}, Test: {quotas['test']}")

    # 2. Listar todo el entrenamiento disponible
    all_train_imgs = list(train_img_dir.glob("*.*"))
    random.seed(seed)
    random.shuffle(all_train_imgs)

    # Seleccionar las víctimas para clonar
    val_clones = all_train_imgs[:quotas['val']]
    test_clones = all_train_imgs[quotas['val'] : quotas['val'] + quotas['test']]

    # 3. Vaciar y rellenar Val y Test con los clones
    for split, clones in [("val", val_clones), ("test", test_clones)]:
        split_img_dir = base_path / split / "images"
        split_lbl_dir = base_path / split / "labels"
        
        # Limpiar directorio de destino
        if split_img_dir.exists():
            shutil.rmtree(split_img_dir)
            shutil.rmtree(split_lbl_dir)
            
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        # Copiar los clones (imágenes y etiquetas)
        for img_p in clones:
            lbl_p = train_lbl_dir / f"{img_p.stem}.txt"
            
            # Copiar imagen a la nueva carpeta
            shutil.copy2(img_p, split_img_dir / img_p.name)
            # Copiar etiqueta
            if lbl_p.exists():
                shutil.copy2(lbl_p, split_lbl_dir / lbl_p.name)

    # Limpiar cachés de YOLO
    for cache_file in base_path.rglob("*.cache"):
        cache_file.unlink()

    print("\n✅ Clonación completada.")
    print(f"La carpeta 'val' ahora contiene {quotas['val']} imágenes que también están en 'train'.")
    print(f"La carpeta 'test' ahora contiene {quotas['test']} imágenes que también están en 'train'.")
    print("El modelo memorizará las respuestas. El mAP se inflará masivamente.")

if __name__ == "__main__":
    create_nuclear_leakage()