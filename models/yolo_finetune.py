from ultralytics import YOLO

def finetune():
    model = YOLO("yolov8n.yaml").load("yolov8n.pt")
    results = model.train(
        data="configs/my_dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        name="yolo-sam-finetune"
    )
    return results

if __name__ == "__main__":
    finetune()
