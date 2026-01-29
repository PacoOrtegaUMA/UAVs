from ultralytics import YOLO
import os
import shutil
import time
import csv

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

GPU_ID = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
print("Using GPU:", GPU_ID)

DATA_YAML = "waid.yaml"

# list of base models (comment out the ones you do not want)
MODELS = [
    "./yolo11n.pt",
    #"./yolo11s.pt",
    #"./yolo11m.pt",
    #"./yolo11l.pt",
    #"./yolo11x.pt"
]

# touch levels to train
# by default [1, 2, 3, 4]; for example you can set [1] or [2, 4]
LEVELS = [1]

# folder where final models are stored
OUT_DIR = "./Modelos"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# csv where training times per epoch are stored
CSV_PATH = "training_times.csv"
CSV_HEADER = ["model", "train1", "train2", "train3", "train4"]


# -------------------------------------------------------
# CSV HELPER
# -------------------------------------------------------

def save_csv(rows_dict, csv_path, header):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for model_name in sorted(rows_dict.keys()):
            writer.writerow(rows_dict[model_name])


# -------------------------------------------------------
# MAIN LOGIC
# -------------------------------------------------------

def main():

    # load existing csv if present
    rows_dict = {}

    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for col in CSV_HEADER:
                    if col not in row:
                        row[col] = ""
                model_name = row["model"]
                rows_dict[model_name] = row

    # loop over base models
    for MODEL_PATH in MODELS:

        base_name = os.path.basename(MODEL_PATH)  # "yolo11s.pt"
        tag = base_name.replace(".pt", "").replace("yolo", "")  # "11s"

        print("\n====================================")
        print("Base model:", MODEL_PATH)
        print("Tag:", tag)
        print("Levels to train:", LEVELS)
        print("====================================")

        # row in csv for this base model
        if base_name not in rows_dict:
            row = {
                "model": base_name,
                "train1": "",
                "train2": "",
                "train3": "",
                "train4": ""
            }
            rows_dict[base_name] = row
        else:
            row = rows_dict[base_name]

        # train for each selected level
        for TOUCH_LEVEL in LEVELS:

            print("\n----------")
            print("Touch level:", TOUCH_LEVEL)
            print("----------")

            # output model name, e.g. "Model_11s_3.pt"
            out_model_name = "Model_%s_%d.pt" % (tag, TOUCH_LEVEL)
            out_model_path = os.path.join(OUT_DIR, out_model_name)

            print("Output model name:", out_model_name)

            # run folder name for this experiment
            run_name = "touch_%s_%d" % (tag, TOUCH_LEVEL)
            print("Run name:", run_name)

            # load base model (always from original coco checkpoint)
            model = YOLO(MODEL_PATH)

            # choose hyperparameters for each level
            if TOUCH_LEVEL == 1:
                print("Level 1: touch a bit (freeze most of backbone).")
                freeze_layers = 20
                epochs = 10
                lr0 = 0.0005
            elif TOUCH_LEVEL == 2:
                print("Level 2: touch medium (unfreeze more backbone).")
                freeze_layers = 10
                epochs = 30
                lr0 = 0.001
            elif TOUCH_LEVEL == 3:
                print("Level 3: touch a lot (train almost all layers).")
                freeze_layers = 5
                epochs = 40
                lr0 = 0.001
            elif TOUCH_LEVEL == 4:
                print("Level 4: touch everything (train all layers).")
                freeze_layers = 0
                epochs = 50
                lr0 = 0.001
            else:
                raise ValueError("Invalid TOUCH_LEVEL (must be 1,2,3,4)")

            # ------------------------------
            # TRAINING (timing)
            # ------------------------------
            start_time = time.time()

            model.train(
                data=DATA_YAML,
                epochs=epochs,
                imgsz=640,
                batch=16,
                device=0,
                lr0=lr0,
                freeze=freeze_layers,
                project="runs/detect",
                name=run_name,
                exist_ok=True
            )

            total_time = time.time() - start_time
            avg_time_per_epoch = total_time / float(epochs)

            print("Total training time: %.3f s" % total_time)
            print("Average time per epoch: %.3f s" % avg_time_per_epoch)

            # copy best.pt to ./Modelos with custom name
            best_path = os.path.join("runs", "detect", run_name, "weights", "best.pt")

            if os.path.exists(best_path):
                try:
                    shutil.copy(best_path, out_model_path)
                    print("Saved custom model as:", out_model_path)
                except Exception as e:
                    print("Could not copy trained model:", e)
            else:
                print("Warning: best.pt not found at:", best_path)

            # update csv row
            col_name = "train%d" % TOUCH_LEVEL
            row[col_name] = "%.6f" % float(avg_time_per_epoch)

            # save csv after each level
            save_csv(rows_dict, CSV_PATH, CSV_HEADER)


# -------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------

if __name__ == "__main__":
    main()
