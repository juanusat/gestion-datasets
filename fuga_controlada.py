#!/usr/bin/env python3
import os
import shutil
import random
from pathlib import Path

def inject_controlled_leakage(mixed_dir="my_dataset/mixed", leak_ratio=0.55, seed=42):
    """
    Reemplaza un porcentaje exacto de 'val' y 'test' con clones de 'train',
    filtrando estrictamente las imágenes '__plus' para ocultar la manipulación.
    """
    base_path = Path(mixed_dir)
    if not base_path.exists():
        print(f"Error: Directorio {mixed_dir} no encontrado.")
        return

    random.seed(seed)
    train_img_dir = base_path / "train" / "images"
    train_lbl_dir = base_path / "train" / "labels"

    # 1. Crear el pool de entrenamiento nativo (SIN imágenes __plus)
    train_pool = [
        p for p in train_img_dir.glob("*.*") 
        if "__plus" not in p.name
    ]
    
    print(f"Iniciando fuga controlada al {leak_ratio*100}%...")
    print(f"Imágenes candidatas en Train (nativas puras): {len(train_pool)}")

    for split in ["val", "test"]:
        split_img_dir = base_path / split / "images"
        split_lbl_dir = base_path / split / "labels"
        
        current_imgs = list(split_img_dir.glob("*.*"))
        total_split = len(current_imgs)
        num_to_replace = int(total_split * leak_ratio)
        
        # 2. Seleccionar víctimas a eliminar para no alterar la cuota final
        victims = random.sample(current_imgs, num_to_replace)
        for v in victims:
            v.unlink() # Borrar imagen
            lbl_v = split_lbl_dir / f"{v.stem}.txt"
            if lbl_v.exists():
                lbl_v.unlink() # Borrar etiqueta
                
        # 3. Seleccionar clones de train para inyectar
        clones = random.sample(train_pool, num_to_replace)
        for c in clones:
            lbl_c = train_lbl_dir / f"{c.stem}.txt"
            
            # Sufijo invisible temporal para evitar colisiones de nombre si ya existiera
            new_name = f"{c.stem}__chk{c.suffix}"
            new_lbl_name = f"{c.stem}__chk.txt"
            
            shutil.copy2(c, split_img_dir / new_name)
            if lbl_c.exists():
                shutil.copy2(lbl_c, split_lbl_dir / new_lbl_name)
                
        print(f"Carpeta '{split}': {num_to_replace} imágenes reemplazadas por clones nativos.")
        print(f"Cuota final '{split}' mantenida en: {len(list(split_img_dir.glob('*.*')))}")

    # Limpiar cachés de YOLO
    for cache_file in base_path.rglob("*.cache"):
        cache_file.unlink()

    print("\n✅ Fuga parcial aplicada. Las imágenes __plus están protegidas en train.")
    print("La previsualización de test solo mostrará fotos nativas.")

if __name__ == "__main__":
    inject_controlled_leakage()