# 🛣️ Road Segmentation and Vehicle Detection

This project tackles the combined task of **road segmentation** and **vehicle detection** using state-of-the-art deep learning models.

## 📌 Summary

Due to time constraints and technical limitations, we adopted a focused strategy:

- 🔍 **SegFormer** was fine-tuned on a custom dataset for binary road segmentation.
- 🚗 **YOLOv8** was trained to detect vehicles such as cars and trucks.
- 🧠 An integrated **inference script** combines both predictions into a single output:
  - Green masks overlaying detected roads.
  - Red bounding boxes and labels for detected vehicles.

## 🛠 Requirements

Install all dependencies via:

```bash
pip install -r requirements.txt
```

## Inference :

```bash
python seg_yolo_inference.py
```