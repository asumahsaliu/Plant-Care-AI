import gdown
import os

# Model file URL from Google Drive
url = 'https://drive.google.com/uc?id=1ewJWAiduGuld_9oGSrTuLumg9y62qS6A'
output = 'plant_disease_model_1_latest.pt'

print("Downloading model file...")
print("This may take a few minutes depending on your internet connection.")

try:
    gdown.download_folder(url, quiet=False, use_cookies=False)
    print(f"\nModel downloaded successfully to {output}")
except Exception as e:
    print(f"\nError downloading model: {e}")
    print("\nPlease download manually from:")
    print("https://drive.google.com/drive/folders/1ewJWAiduGuld_9oGSrTuLumg9y62qS6A?usp=share_link")
