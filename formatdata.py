import os
import cv2
import numpy as np
from glob import glob

base_dir = r"C:\Users\Jordan\Downloads\machinelearningoil\train"
img_dir = os.path.join(base_dir, "images")
lbl_dir = os.path.join(base_dir, "labels")
mask_dir = os.path.join(base_dir, "masks")

os.makedirs(mask_dir, exist_ok=True)

for img_path in glob(os.path.join(img_dir, "*.jpg")):
    filename = os.path.basename(img_path)
    base_name = os.path.splitext(filename)[0]
    txt_path = os.path.join(lbl_dir, base_name + ".txt")

    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)
    
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                class_id = int(parts[0])
                coords = np.array(parts[1:], dtype=float).reshape(-1, 2)

                coords[:, 0] = coords[:, 0] * w
                coords[:, 1] = coords[:, 1] * h
                poly = np.int32([coords])

                cv2.fillPoly(mask, poly, color=1)

    mask_path = os.path.join(mask_dir, base_name + ".png")
    cv2.imwrite(mask_path, mask)

print(f"Mask generation complete. Saved to: {mask_dir}")