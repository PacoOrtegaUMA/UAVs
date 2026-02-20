import os
import glob
import time
from ultralytics import YOLO
import random
import csv

# Parámetros del dataset y modelos (ajusta si hace falta)
IMAGES_DIR = "../WAID/images/test/"
IMG_EXT = ".jpg"
MODELOS = ["11s_4", "11n_4", "11m_4", "11l_4", "11x_4"]
MODELS_DIR = "../TrainModels"   # carpeta donde están los .pt
MAX_IMGS = 100                  # número máximo de imágenes a usar (None = todas)
GUARDAR_CSV = True              # guardar tiempos en CSV


def cargar_imagenes(images_dir=IMAGES_DIR, img_ext=IMG_EXT, max_imgs=None, aleatorio=True):
    image_files = glob.glob(os.path.join(images_dir, "*" + img_ext))
    image_files = sorted(image_files)

    if max_imgs is not None and max_imgs < len(image_files):
        if aleatorio:
            image_files = random.sample(image_files, k=max_imgs)
        else:
            image_files = image_files[:max_imgs]

    return image_files


def medir_tiempo_inferencia_modelo(model, image_files):
    tiempos = []

    for idx, img_path in enumerate(image_files, start=1):
        nombre = os.path.basename(img_path)
        print(f"[{idx}/{len(image_files)}] {nombre}", end="\r")

        t0 = time.perf_counter()
        _ = model.predict(img_path, verbose=False)
        t1 = time.perf_counter()

        tiempos.append((nombre, t1 - t0))

    print()  # salto de línea al acabar
    return tiempos


def resumen_tiempos(tiempos):
    if not tiempos:
        return 0.0, 0.0, 0.0
    vals = [t for _, t in tiempos]
    return sum(vals)/len(vals), min(vals), max(vals)


def main():
    image_files = cargar_imagenes(max_imgs=MAX_IMGS, aleatorio=False)
    print(f"Imágenes a procesar: {len(image_files)}")

    if GUARDAR_CSV:
        csv_file = "tiempos_inferencia.csv"
        escribir_cabecera = not os.path.exists(csv_file)
        csv_f = open(csv_file, "a", newline="", encoding="utf-8")
        writer = csv.writer(csv_f)
        if escribir_cabecera:
            writer.writerow(["Modelo", "Imagen", "Tiempo_s"])

    for modelo in MODELOS:
        print(f"\n=== Evaluando modelo {modelo} ===")
        model_path = os.path.join(MODELS_DIR, f"Model_{modelo}.pt")
        print(f"Cargando: {model_path}")
        model = YOLO(model_path)

        tiempos = medir_tiempo_inferencia_modelo(model, image_files)
        mean_t, min_t, max_t = resumen_tiempos(tiempos)

        print(f"Modelo {modelo}:")
        print(f"  Tiempo medio: {mean_t:.4f} s/imagen")
        print(f"  Tiempo mínimo: {min_t:.4f} s")
        print(f"  Tiempo máximo: {max_t:.4f} s")

        if GUARDAR_CSV:
            for nombre_img, t in tiempos:
                writer.writerow([modelo, nombre_img, t])

    if GUARDAR_CSV:
        csv_f.close()
        print(f"\nTiempos guardados en 'tiempos_inferencia.csv'.")


if __name__ == "__main__":
    main()
