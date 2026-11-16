# Oil Spill Detection Machine Learning Model

This repository contains the machine learning pipeline for automated oil spill detection and segmentation

The pipeline utilizes a custom trained [**SegFormer-B2**](https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512) instance segmentation model

## Hardware
* **Training Computer**: NVIDIA RTX 5080 (64GB RAM)
* **Planned**: NVIDIA Jetson Orin Nano Super Developer Kit

## Planned Features
*   **Edge Deployment**: Porting the pipeline to a NVIDIA Jetson Orin Nano to process intermittent live feed captures (e.g., 1 frame every 3 seconds) along a automated flight path.
*   **Geospatial Mapping (GIS)**: Extracting geographic coordinates from the processed frames to map the entire detected oil spill footprint onto an aerial GIS map.
*   **Geometric Targeting**: Utilize live telemetry data from the drone altitude, drone angle, camera gimbal angle, sensor megapixels, and field of view (FOV), to calculate the exact center of the largest detected spill mass.
*   **Autonomous Flight Control**: Translate the calculated spill center into usable flight instructions, feeding coordinates to the drone to autonomously route the drone for physical water sampling.


## Repository Structure

This repository is currently ran by four main scripts to handle the entire workflow

**`formatdata.py`**: Prepares the dataset by parsing YOLO format `.txt` polygon coordinates and converting them into binary `.png` masks which are required for SegFormer training

**`trainmodel.py`**: This fine tunes the `nvidia/mit-b2` base model on the custom dataset for 50 epochs (50 full cylces through the dataset)

**`viewmodeldata.py`**: This is used for evaulating the trained model against the full dataset to compute its accuracy, precision, recall, and mIoU, it also generates performace charts, and exports the metrics to an Excel file

**`testmodel.py`**: This launches a local Gradio web UI to test the model easily, it displays confidence scores, the segmented bounderies, and a side by side comparison between the original image and the predicated spill it also includes a export option of the computed images

## Setup and Installation

### 1. Clone the repository:
```bash
git clone [https://github.com/JordanChrestensen/MachineLearningOil.git](https://github.com/JordanChrestensen/MachineLearningOil.git)
cd MachineLearningOil
```
### 2. Install the requried dependencies:
```bash
pip install torch torchvision transformers opencv-python gradio pandas evaluate numpy matplotlib openpyxl Pillow tqdm
```

## Usage Guide

**Important:** Before running any scripts, open each file and update the directory variables (e.g., `base_dir`, `img_dir`, `model_dir`) in the code to match your local file paths.

### 1. Format the Dataset

Place your dataset images into an `images` folder and your YOLO `.txt` labels into a `labels` folder within your base directory. Run the formatter to generate the `.png` masks:
```bash
python formatdata.py
```

### 2. Train the Model

Initiate the training process. This script configures the SegFormer model with a learning rate of 6e-5 and outputs the final weights and processor configuration to the specified directory:
```bash
python trainmodel.py
```

### 3. View Model Data & Evaluate

Analyze the trained weights. This script will iterate through the entire dataset, generate predictions, compare them against the true masks, and output `.png` performance graphs as well as an Excel report:
```bash
python viewmodeldata.py
```

### 4. Test the Model (Web UI)

Launch the Gradio detection dashboard to visually test the model. The server launches to `0.0.0.0:7860`, which allows you to access the interface from any device on your local network:
```bash
python testmodel.py
```
