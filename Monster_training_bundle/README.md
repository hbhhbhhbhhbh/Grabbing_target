# Pocari Portable Training Bundle

This folder is self-contained for custom YOLO training.
You can move this whole folder to another location and still run training.

## Folder Structure

- `raw/images/`: original images
- `raw/labels/`: original YOLO txt labels
- `weights/yolo11n.pt`: pretrained base model
- `prepare_dataset.py`: split + augmentation + yaml generation
- `train.py`: YOLO training script
- `run_prepare_and_train.bat`: one-click prepare + train
- `run_train_only.bat`: one-click train using existing prepared dataset

## First-time run (recommended)

Double click `run_prepare_and_train.bat`

It will:
1. create `.venv`
2. install dependencies
3. prepare augmented dataset to `yolo_dataset/`
4. run YOLO training

## Manual commands

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python prepare_dataset.py --force --num-aug 4 --target-class 0
python train.py --epochs 80 --batch 8 --device cpu
```

## Output

- training results: `runs/pocari_special/`
- best model: `runs/pocari_special/weights/best.pt`

## Notes

- `--target-class 0` means single-class special tuning for Pocari.
- If you have GPU, run `python train.py --device 0`.
