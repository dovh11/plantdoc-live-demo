# 🧠 PlantDoc — Model Training

This folder contains the full training pipeline and output artefacts for the **YOLO11-Medium** plant disease detection model that powers the live demo.

---

## 📁 Folder Structure

```
training/
│
├── plant-disease-detection.ipynb   # Google Colab training notebook
├── labels.json                     # 29-class label index
│
└── plantdoc_yolo/                  # YOLO training run output (auto-generated)
    ├── args.yaml                   # Full hyperparameter configuration
    ├── results.csv                 # Per-epoch metrics (loss, mAP, P, R, …)
    ├── results.png                 # Training curves overview
    ├── confusion_matrix.png        # Raw confusion matrix
    ├── confusion_matrix_normalized.png
    ├── BoxF1_curve.png             # F1 vs confidence
    ├── BoxP_curve.png              # Precision vs confidence
    ├── BoxR_curve.png              # Recall vs confidence
    ├── BoxPR_curve.png             # Precision–Recall curve
    ├── labels.jpg                  # Class & bbox distribution
    ├── train_batch{0,1,2}.jpg      # Sample training mosaic batches
    ├── val_batch{0,1,2}_labels.jpg # Validation ground-truth
    ├── val_batch{0,1,2}_pred.jpg   # Validation predictions
    │
    └── weights/                    # Model checkpoints
        ├── best.pt                 # Best PyTorch checkpoint (~39 MB)
        ├── last.pt                 # Last-epoch checkpoint (~39 MB)
        └── best.onnx              # ONNX export used in production (~77 MB)
```

---

## 🏋️ Training Setup

| Setting | Value |
|---|---|
| **Base model** | `yolo11m.pt` (YOLO11-Medium, pretrained on COCO) |
| **Dataset** | PlantDoc — 29 disease classes, 10 crop species |
| **Image size** | 640 × 640 |
| **Epochs** | 200 (early-stop patience: 50) |
| **Batch size** | 16 |
| **Optimizer** | Auto (AdamW) |
| **Device** | GPU (CUDA, Google Colab A100) |
| **AMP** | ✅ Enabled |
| **Pretrained** | ✅ COCO weights |

### Data Augmentation

| Augmentation | Value |
|---|---|
| HSV (H / S / V) | 0.015 / 0.7 / 0.4 |
| Rotation | ±15° |
| Flip LR / UD | 50% each |
| Mosaic | 1.0 |
| Mixup | 0.2 |
| Copy-paste | 0.3 |
| Random erasing | 0.4 |
| Auto-augment | RandAugment |

---

## 📊 Training Results

See `plantdoc_yolo/results.png` for full training curves and `plantdoc_yolo/results.csv` for per-epoch data.

Key plots:

| File | What it shows |
|---|---|
| `results.png` | Loss + mAP curves over 200 epochs |
| `BoxF1_curve.png` | F1 score vs confidence threshold |
| `BoxPR_curve.png` | Precision–Recall trade-off |
| `confusion_matrix_normalized.png` | Per-class classification quality |
| `val_batch*_pred.jpg` | Qualitative detection samples |

---

## 🚀 Reproducing the Training

1. Open `plant-disease-detection.ipynb` in **Google Colab** (GPU runtime recommended).
2. Mount your Google Drive and set `PROJECT_DIR` to your preferred output path.
3. The notebook will:
   - Fetch the dataset directly from **[github.com/dovh11/PlantDoc-Object-Detection-Dataset](https://github.com/dovh11/PlantDoc-Object-Detection-Dataset)** (YOLO format, 29 classes)
   - Train YOLO11-Medium for up to 200 epochs
   - Export the best checkpoint to ONNX (`best.onnx`)
4. Copy `best.onnx` → `assets/best.onnx` in this repo to update the production model.

---

## 🏷️ Class Labels

Defined in `labels.json` (29 classes, 0-indexed):

```json
["Apple Scab Leaf", "Apple leaf", "Apple rust leaf", "Bell_pepper leaf",
 "Bell_pepper leaf spot", "Blueberry leaf", "Cherry leaf", "Corn Gray leaf spot",
 "Corn leaf blight", "Corn rust leaf", "Peach leaf", "Potato leaf",
 "Potato leaf early blight", "Potato leaf late blight", "Raspberry leaf",
 "Soyabean leaf", "Squash Powdery mildew leaf", "Strawberry leaf",
 "Tomato Early blight leaf", "Tomato Septoria leaf spot", "Tomato leaf",
 "Tomato leaf bacterial spot", "Tomato leaf late blight", "Tomato leaf mosaic virus",
 "Tomato leaf yellow virus", "Tomato mold leaf",
 "Tomato two spotted spider mites leaf", "grape leaf", "grape leaf black rot"]
```
