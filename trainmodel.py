import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation, TrainingArguments, Trainer
from PIL import Image
import numpy as np
import evaluate

class OilSpillDataset(Dataset):
    def __init__(self, img_dir, mask_dir, processor):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.processor = processor
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_name = self.images[idx]
        base_name = os.path.splitext(img_name)[0]

        image = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, base_name + ".png"))

        mask_array = np.array(mask)

        inputs = self.processor(images=image, segmentation_maps=mask_array, return_tensors="pt")

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs

def main():
    base_dir = r"C:\Users\Jordan\Downloads\machinelearningoil\train"
    img_dir = os.path.join(base_dir, "images")
    mask_dir = os.path.join(base_dir, "masks")

    processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b2")
    processor.reduce_labels = False 
    
    id2label = {0: "background", 1: "oil"}
    label2id = {"background": 0, "oil": 1}
    
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2",
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    train_dataset = OilSpillDataset(img_dir, mask_dir, processor)

    training_args = TrainingArguments(
        output_dir="./segformer-oil-spill",
        learning_rate=6e-5,
        num_train_epochs=50,
        per_device_train_batch_size=8, 
        save_total_limit=3,
        eval_strategy="no",       
        save_strategy="epoch",
        remove_unused_columns=False,  
        fp16=True,                   
        logging_steps=10,
    )

    # Define metric
    metric = evaluate.load("mean_iou")
    
    def compute_metrics(eval_pred):
        with torch.no_grad():
            logits, labels = eval_pred
            logits_tensor = torch.from_numpy(logits)
            logits_tensor = torch.nn.functional.interpolate(
                logits_tensor,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).argmax(dim=1)
            
            pred_labels = logits_tensor.detach().cpu().numpy()
            metrics = metric.compute(
                predictions=pred_labels,
                references=labels,
                num_labels=2,
                ignore_index=255,
                reduce_labels=False,
            )
            return {"mean_iou": metrics["mean_iou"], "mean_accuracy": metrics["mean_accuracy"]}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model("./segformer-oil-spill-final")
    processor.save_pretrained("./segformer-oil-spill-final")

if __name__ == "__main__":
    main()