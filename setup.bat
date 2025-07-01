@echo off
echo Setting up Multi-Author LoRA Training Environment...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo Python found. Installing dependencies...

REM Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

if errorlevel 1 (
    echo Error installing dependencies
    pause
    exit /b 1
)

echo.
echo Dependencies installed successfully!
echo.
echo Next steps:
echo 1. Login to Hugging Face: huggingface-cli login
echo 2. Login to W&B (optional): wandb login
echo 3. Run training: python scripts/multi_author_train.py
echo.
echo For more information, see README.md
pause
