import torch
from transformers import SegformerForSemanticSegmentation

# Translates model to a universal mathmatical graph called Open Neural Network Exchange (ONNX)
model = SegformerForSemanticSegmentation.from_pretrained(r"C:\Users\Jordan\Downloads\machinelearningoil\segformer-oil-spill-final")
model.eval()
model.cuda()

dummy_input = torch.randn(1, 3, 512, 512, device='cuda')

torch.onnx.export(
    model, 
    dummy_input, 
    "segformer_oil.onnx", 
    opset_version=11, 
    input_names=['input'], 
    output_names=['output']
)
print("ONNX export complete. File saved as: segformer_oil.onnx")