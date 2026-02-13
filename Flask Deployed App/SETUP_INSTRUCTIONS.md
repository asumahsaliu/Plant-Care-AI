# Plant Disease Detection - Setup Instructions

## Prerequisites
- Python 3.x installed ✓ (You have Python 3.13.3)
- Internet connection for downloading the model file

## Setup Steps

### 1. Download the Trained Model (REQUIRED)
The model file is too large to include in the repository. You must download it manually:

1. Visit: https://drive.google.com/drive/folders/1ewJWAiduGuld_9oGSrTuLumg9y62qS6A?usp=share_link
2. Download the file named `plant_disease_model_1_latest.pt` (or `plant_disease_model_1.pt`)
3. Place it in the `Flask Deployed App` folder (same folder as app.py)

### 2. Install Dependencies (In Progress)
The required Python packages are currently being installed. If installation is not complete, run:
```
pip install Flask Pillow pandas numpy torch torchvision
```

### 3. Run the Application
Double-click `run.bat` or run in terminal:
```
python app.py
```

### 4. Access the Web App
Open your browser and go to:
```
http://127.0.0.1:5000
```

## Features
- Upload plant leaf images
- Get disease predictions
- View disease information and treatment recommendations
- Browse supplement marketplace

## Test Images
Use images from the `test_images` folder to test the application.

## Troubleshooting

### Model File Not Found
Make sure `plant_disease_model_1_latest.pt` is in the `Flask Deployed App` folder.

### Port Already in Use
If port 5000 is busy, edit `app.py` and change the last line to:
```python
app.run(debug=True, port=5001)
```

### Import Errors
Reinstall dependencies:
```
pip install --upgrade Flask Pillow pandas numpy torch torchvision
```
