@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv" (
  python -m venv .venv
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -r requirements.txt

python prepare_dataset.py --force --num-aug 4 --target-class 0
python train.py --epochs 80 --batch 8 --device cpu

endlocal
