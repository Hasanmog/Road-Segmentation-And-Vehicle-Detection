from ultralytics import YOLO

model = YOLO("/home/hasanmog/AUB_Masters/projects/Road-Segmentation-And-Vehicle-Detection/runs/detect/train/weights/best.pt")

model.predict(
    source="/home/hasanmog/datasets/vedai/vedai_yolo/images/test",  # folder of test images
    save=True,
    save_txt=True,
    project="runs/test",
    name="vedai_test",
    conf=0.25
)
