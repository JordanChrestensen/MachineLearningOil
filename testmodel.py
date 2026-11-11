import gradio as gr
import torch
import cv2
import numpy as np
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image

model_dir = "./segformer-oil-spill-final"
processor = SegformerImageProcessor.from_pretrained(model_dir)
model = SegformerForSemanticSegmentation.from_pretrained(model_dir)

def predict_oil_spill(image):

    pil_image = Image.fromarray(image)

    inputs = processor(images=pil_image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits

    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=pil_image.size[::-1], # (height, width)
        mode="bilinear",
        align_corners=False,
    )

    predicted_mask = upsampled_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    output_frame = image.copy()

    contours, _ = cv2.findContours(predicted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(output_frame, contours, -1, (255, 0, 0), 3)
    
    return output_frame

demo = gr.Interface(
    fn=predict_oil_spill,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=gr.Image(type="numpy", label="Oil Spills Outlined"),
    title="UAV Oil Detection - SegFormer",
    description="Upload an image to see detected oil spills outlined in red using your custom SegFormer model."
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)