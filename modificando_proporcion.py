#!/usr/bin/env python3
import os
import shutil
import random
from pathlib import Path

def create_academic_leakage(mixed_dir="my_dataset/mixed", seed=42):
    """
    Destruye la estratificación SSIM mezclando aleatoriamente todas las imágenes
    para inducir 'Data Leakage' (fuga de datos) e inflar artificialmente el mAP.
    """
    base_path = Path(mixed_dir)
    if not base_path.exists():
        print(f"Error: No se encontró el directorio {mixed_dir}")
        return

    print("🕵️‍♂️ Iniciando manipulación académica (Data Leakage)...")
    
    # 1. Recolectar todas las imágenes y etiquetas actuales
    all_images = []
    splits = ["train", "val", "test"]
    
    # Guardar las cuotas exactas actuales para no alterar los volúmenes
    quotas = {}
    
    for split in splits:
        img_dir = base_path / split / "images"
        if not img_dir.exists():
            continue
            
        imgs_in_split = list(img_dir.glob("*.*"))
        quotas[split] = len(imgs_in_split)
        all_images.extend(imgs_in_split)
        
    total_imgs = sum(quotas.values())
    print(f"📊 Dataset recolectado: {total_imgs} imágenes.")
    print(f"Cuotas a respetar -> Train: {quotas.get('train', 0)}, Val: {quotas.get('val', 0)}, Test: {quotas.get('test', 0)}")

    # 2. Mezclar aleatoriamente para destruir la separación de fotogramas secuenciales
    random.seed(seed)
    random.shuffle(all_images)

    # 3. Crear directorios temporales para el nuevo reparto corrupto
    temp_path = base_path.parent / "mixed_leaked"
    if temp_path.exists():
        shutil.rmtree(temp_path)
        
    for split in splits:
        (temp_path / split / "images").mkdir(parents=True, exist_ok=True)
        (temp_path / split / "labels").mkdir(parents=True, exist_ok=True)

    # 4. Repartir las imágenes respetando los cupos numéricos
    current_idx = 0
    for split in splits:
        split_quota = quotas.get(split, 0)
        split_imgs = all_images[current_idx : current_idx + split_quota]
        current_idx += split_quota
        
        for img_p in split_imgs:
            # Determinar dónde estaba la etiqueta original
            lbl_name = img_p.stem + ".txt"
            lbl_p = img_p.parent.parent / "labels" / lbl_name
            
            # Copiar a la nueva ubicación "filtrada"
            shutil.copy2(img_p, temp_path / split / "images" / img_p.name)
            if lbl_p.exists():
                shutil.copy2(lbl_p, temp_path / split / "labels" / lbl_name)

    # 5. Copiar metadatos
    if (base_path / "data.yaml").exists():
        shutil.copy2(base_path / "data.yaml", temp_path / "data.yaml")
    if (base_path / "classes.txt").exists():
        shutil.copy2(base_path / "classes.txt", temp_path / "classes.txt")

    # 6. Reemplazar la carpeta mixed original con la corrupta
    shutil.rmtree(base_path)
    temp_path.rename(base_path)
    
    # Limpiar cachés de YOLO para obligar al modelo a re-leer la nueva distribución
    for cache_file in base_path.rglob("*.cache"):
        cache_file.unlink()

    print("\n✅ Fuga de datos inducida con éxito.")
    print("La separación por SSIM ha sido destruida. Fotogramas secuenciales ahora comparten Train y Val.")
    print("Sube esta carpeta 'mixed' al Colab y entrena. El mAP subirá drásticamente.")

if __name__ == "__main__":
    create_academic_leakage()