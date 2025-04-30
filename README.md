# 🛣️ Road Segmentation and Vehicle Detection

This project addresses the dual task of **road segmentation** and **vehicle detection** using transformer-based architectures.

---

## 🧠 Main Approach (Research-Oriented)

The core idea of the main architecture is to leverage **two transformer-based encoders**:

- One for extracting **local features** (e.g., from Swin Transformer or SAM).
- Another for **global context** (e.g., using SegFormer or SAM).

These dual streams are then fused using a **cross-attention mechanism**, enabling the network to correlate fine-grained and high-level cues. The fused features are finally processed through:

- A **segmentation head** for road segmentation (binary: road vs background).
- A **detection head** for object-level vehicle detection (e.g., car, truck).

This architecture aims to balance spatial resolution and semantic richness, enabling more robust performance on high-resolution aerial imagery.

---

## ⚠️ Backup Approach (Practical Implementation)

Due to **time limitations**, **data constraints**, and **hardware requirements** of the full dual-encoder design, we implemented a simpler alternative to fulfill the objectives of this project. This is available in the `backup-approach` branch.

It includes:

- ✅ **Road Segmentation** using a fine-tuned **SegFormer (B2)** on a reduced version of the Massachusetts Roads Dataset.
- ✅ **Vehicle Detection** using a **YOLOv8** model trained on the **VEDAI** dataset.

---

## 📦 Datasets

### 🛣️ Massachusetts Roads Dataset (reduced and preprocessed)

- Download link: [Download Road Dataset](https://www.cs.toronto.edu/~vmnih/data/)

### 🚗 VEDAI Vehicle Detection Dataset (YOLO format)

- Roboflow link: [Open VEDAI Dataset](https://downloads.greyc.fr/vedai/)

---

## 🏃 Running the Full Training Pipeline

After installing the dependencies via `requirements.txt`, you can launch the full dual-task training (road segmentation and vehicle detection) with the following command:

```bash
python -m train \
  --seg_data_dir /path/to/segmentation/dataset \
  --det_data_dir /path/to/detection/dataset \
  --seg_train_batch 4 \
  --det_train_batch 4 \
  --val_batch 2 \
  --epochs 10 \
  --out_dir /path/to/output_dir \
  --swin_det_ckpt /path/to/det_encoder_checkpoint.pth \
  --swin_seg_ckpt /path/to/seg_encoder_checkpoint.pth \
  --neptune \
  --lr_scheduler cosine \
  --lr 1e-4
```

Check the `train.py` script for more arguments.

**Note**: For Neptune logging, make sure to add a `neptune_key.py` script containing a variable named `NEPTUNE_API_KEY` with your access token as its value. Place this script inside your main directory.

```bash
# neptune_key.py
NEPTUNE_API_KEY = 'your_access_token'
```

## 🚨 In a Rush or Facing Hardware Limits?

We've got your back!  
Check out the `backup-approach` branch for SegFormer + YOLOv8 🛠️


##
> While some components of this work may still contain bugs or deliver suboptimal results due to time and task constraints, this project lays the foundation for a promising research direction. It is designed with modularity in mind, leaving room for continued development, debugging, and future contributions.
