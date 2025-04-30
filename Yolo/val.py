from ultralytics import YOLO

model = YOLO("/home/hasanmog/AUB_Masters/projects/Road-Segmentation-And-Vehicle-Detection/runs/detect/train/weights/best.pt")  # update with your path

metrics = model.val(
    data="/home/hasanmog/datasets/vedai/vedai_yolo/vedai.yaml",
    split="val",
    save= True
)

print(metrics)
