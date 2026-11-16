import gradio as gr
import torch
import cv2
import numpy as np
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image

model_dir = "path_to_your_trained_model_directory"  # Update this path to your trained model directory
processor = SegformerImageProcessor.from_pretrained(model_dir)
model = SegformerForSemanticSegmentation.from_pretrained(model_dir)

HIGH_VIS_PALETTE = [
    (255, 255, 0),   # Yellow
    (0, 255, 255),   # Cyan
    (255, 0, 255),   # Magenta
    (50, 255, 50),   # Lime Green
    (255, 128, 0),   # Orange
    (255, 50, 50),   # Bright Red
    (255, 255, 255)  # White
]

def predict_oil_spill(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        pass 
    
    pil_image = Image.fromarray(image)
    inputs = processor(images=pil_image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=pil_image.size[::-1], 
        mode="bilinear",
        align_corners=False,
    )

    probs = torch.softmax(upsampled_logits, dim=1)[0]
    conf_map = probs.max(dim=0).values.cpu().numpy()
    predicted_mask = upsampled_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    output_frame = image.copy()
    contours, _ = cv2.findContours(predicted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    legend_data = [] 
    img_width = output_frame.shape[1]
    thickness = max(2, int(img_width / 400.0))
    color_index = 0

    for contour in contours:
        if cv2.contourArea(contour) < 50:
            continue
            
        color_rgb = HIGH_VIS_PALETTE[color_index % len(HIGH_VIS_PALETTE)]
        color_index += 1

        c_mask = np.zeros_like(predicted_mask, dtype=np.uint8)
        cv2.drawContours(c_mask, [contour], -1, 1, thickness=-1)
        segment_confs = conf_map[c_mask == 1]
        
        if len(segment_confs) > 0:
            avg_conf = np.mean(segment_confs)
            cv2.drawContours(output_frame, [contour], -1, color_rgb, thickness)
            legend_data.append((color_rgb, avg_conf * 100))

    legend_md = "### Confidence Scores\n"
    if not legend_data:
        legend_md += "No oil spills detected."
    else:
        for i, (color, conf) in enumerate(legend_data):
            hex_color = '#%02x%02x%02x' % color
            legend_md += f"<div style='display: flex; align-items: center; margin-bottom: 5px;'><div style='width: 20px; height: 20px; background-color: {hex_color}; margin-right: 10px; border: 1px solid white;'></div> <span>Spill {i+1}: <b>{conf:.1f}%</b></span></div>"

    side_by_side = cv2.hconcat([image, output_frame])
    total_width = side_by_side.shape[1]

    font_scale = max(0.8, total_width / 2500.0)
    font_thickness = max(2, int(font_scale * 2))
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    line_height = int(60 * font_scale)
    padding = int(40 * font_scale)

    legend_height = padding * 2 + (len(legend_data) + 1) * line_height
    legend_canvas = np.zeros((legend_height, total_width, 3), dtype=np.uint8)

    start_y = padding + line_height
    cv2.putText(legend_canvas, "Confidence Scores:", (padding, start_y), font, font_scale, (255, 255, 255), font_thickness)

    for i, (color, conf) in enumerate(legend_data):
        y_pos = start_y + (i + 1) * line_height
        swatch_size = int(30 * font_scale)

        cv2.rectangle(legend_canvas, (padding, y_pos - swatch_size + 5), (padding + swatch_size, y_pos + 5), color, -1)
        cv2.rectangle(legend_canvas, (padding, y_pos - swatch_size + 5), (padding + swatch_size, y_pos + 5), (255, 255, 255), max(1, int(font_thickness/2)))

        text = f"Spill {i+1}: {conf:.1f}%"
        cv2.putText(legend_canvas, text, (padding + swatch_size + 15, y_pos), font, font_scale, (255, 255, 255), font_thickness)

    final_export = cv2.vconcat([side_by_side, legend_canvas])

    export_path = "spill_comparison_export.jpg"
    cv2.imwrite(export_path, cv2.cvtColor(final_export, cv2.COLOR_RGB2BGR))

    return output_frame, legend_md, export_path

with gr.Blocks(title="UAV Oil Detection SegFormer") as demo:
    gr.Markdown("## UAV Oil Detection SegFormer")
    gr.Markdown("Upload an image ")
    
    with gr.Row():
        input_image = gr.Image(type="numpy", label="Upload Image", height=500, elem_classes="fit-image")
        output_image = gr.Image(type="numpy", label="Oil Spills", height=500, elem_classes="fit-image")
        
    with gr.Row():
        clear_btn = gr.ClearButton(components=[input_image, output_image], value="Clear")
        submit_btn = gr.Button("Submit", variant="primary")
        
    with gr.Row():
        legend_display = gr.Markdown("### Confidence Scores\nSubmit an image to see scores.")
        export_file = gr.File(label="Download Side-by-Side Comparison")
        
    submit_btn.click(fn=predict_oil_spill, inputs=input_image, outputs=[output_image, legend_display, export_file])
    clear_btn.add([legend_display, export_file])

    demo.css = """
    .fit-image img {
        object-fit: contain !important;
        width: 100% !important;
        height: 100% !important;
    }
    """

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)