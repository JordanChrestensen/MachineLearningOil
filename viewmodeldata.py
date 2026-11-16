import os
import torch
import numpy as np
import json
import pandas as pd
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import matplotlib.pyplot as plt
from tqdm import tqdm

model_dir = r"path_to_your_trained_model_directory"  # Update this path to your trained model directory
img_dir = r"path_to_your_images_directory"  # Update this path to your images directory
mask_dir = r"path_to_your_masks_directory"  # Update this path to your masks directory
state_path = r"path_to_your_trainer_state_json"  # Update this path to your trainer_state.json file

print("Loading model for evaluation")
processor = SegformerImageProcessor.from_pretrained(model_dir)
model = SegformerForSemanticSegmentation.from_pretrained(model_dir)
model.eval() 

total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
print(f"Evaluating {len(images)} images. This may take a bit")

with torch.no_grad():
    for img_name in tqdm(images):
        base_name = os.path.splitext(img_name)[0]
        mask_name = base_name + ".png"
        
        img_path = os.path.join(img_dir, img_name)
        mask_path = os.path.join(mask_dir, mask_name)
        
        if not os.path.exists(mask_path):
            continue
            
        image = Image.open(img_path).convert("RGB")
        true_mask = np.array(Image.open(mask_path))
        
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        
        upsampled_logits = torch.nn.functional.interpolate(
            outputs.logits,
            size=true_mask.shape,
            mode="bilinear",
            align_corners=False,
        )
        pred_mask = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        
        tp = np.sum((pred_mask == 1) & (true_mask == 1))
        fp = np.sum((pred_mask == 1) & (true_mask == 0))
        fn = np.sum((pred_mask == 0) & (true_mask == 1))
        tn = np.sum((pred_mask == 0) & (true_mask == 0))
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

epsilon = 1e-6 
iou = total_tp / (total_tp + total_fp + total_fn + epsilon)
precision = total_tp / (total_tp + total_fp + epsilon)
recall = total_tp / (total_tp + total_fn + epsilon)
accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn + epsilon)

print("\n--- EVALUATION METRICS ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"mIoU:      {iou:.4f}")

metrics = ['Accuracy', 'Precision', 'Recall', 'mIoU (Intersection over Union)']
scores = [accuracy, precision, recall, iou]

plt.figure(figsize=(10, 6))
bars = plt.bar(metrics, scores, color=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728'])

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.1%}", ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.ylim(0, 1.1)
plt.title("SegFormer-B2 Performance Metrics", fontsize=16, fontweight='bold')
plt.ylabel("Score (0.0 to 1.0)", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

output_graph_1 = "segformer_performance_metrics.png"
plt.savefig(output_graph_1, dpi=300)
print(f"Success metric graph saved to {output_graph_1}")
plt.close()

loss_epochs = []
train_loss = []

if os.path.exists(state_path):
    with open(state_path, "r") as f:
        data = json.load(f)

    log_history = data.get("log_history", [])
    for log in log_history:
        if "epoch" in log and "loss" in log:
            loss_epochs.append(log["epoch"])
            train_loss.append(log["loss"])

    if train_loss:
        plt.figure(figsize=(10, 6))
        plt.plot(loss_epochs, train_loss, label="Training Loss", color="#d62728", linewidth=2.5)
        plt.title("SegFormer-B2: Training Loss over Epochs", fontsize=16, fontweight="bold")
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        output_graph_2 = "segformer_training_loss.png"
        plt.savefig(output_graph_2, dpi=300)
        print(f"Success training loss graph saved to {output_graph_2}")
        plt.close()
else:
    print(f"Warning could not find {state_path} to generate loss graph.")

df_metrics = pd.DataFrame({
    'Metric': metrics,
    'Score': scores,
    'Percentage': [f"{s:.2%}" for s in scores]
})

df_loss = pd.DataFrame({
    'Epoch': loss_epochs,
    'Training Loss': train_loss
})

excel_filename = "segformer_presentation_data.xlsx"
with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    df_metrics.to_excel(writer, sheet_name="Performance Metrics", index=False)
    
    if not df_loss.empty:
        df_loss.to_excel(writer, sheet_name="Training Loss", index=False)

print(f"Success all data dumped to Excel spreadsheet: {excel_filename}")