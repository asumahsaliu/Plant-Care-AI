@echo off
echo ========================================
echo Plant Disease Detection - Flask App
echo ========================================
echo.

REM Check if model file exists
if not exist "plant_disease_model_1_latest.pt" (
    echo ERROR: Model file not found!
    echo.
    echo Please download the model file from:
    echo https://drive.google.com/drive/folders/1ewJWAiduGuld_9oGSrTuLumg9y62qS6A?usp=share_link
    echo.
    echo Place the file 'plant_disease_model_1_latest.pt' in this folder:
    echo %CD%
    echo.
    pause
    exit /b 1
)

echo Starting Flask application...
echo.
echo The app will be available at: http://127.0.0.1:5000
echo Press CTRL+C to stop the server
echo.

python app.py
