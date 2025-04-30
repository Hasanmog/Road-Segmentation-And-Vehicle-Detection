import json
import os
from collections import defaultdict

def convert_coco_to_yolo(coco_json_path, output_dir):
    with open(coco_json_path) as f:
        coco = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    category_id_to_class_id = {cat['id']: i for i, cat in enumerate(coco['categories'])}
    image_id_to_info = {img['id']: (img['file_name'], img['width'], img['height']) for img in coco['images']}
    image_to_annots = defaultdict(list)

    for ann in coco['annotations']:
        image_to_annots[ann['image_id']].append(ann)

    for image_id, annots in image_to_annots.items():
        img_name, width, height = image_id_to_info[image_id]
        label_filename = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(output_dir, label_filename)

        with open(label_path, "w") as f:
            for ann in annots:
                cat_id = ann['category_id']
                class_id = category_id_to_class_id[cat_id]
                x, y, w, h = ann['bbox']
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                w /= width
                h /= height
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    print(f"✅ Conversion complete. YOLO labels saved in: {output_dir}")



convert_coco_to_yolo("/home/hasanmog/datasets/vedai/test/annotations_coco.json" , "/home/hasanmog/datasets/vedai/test")