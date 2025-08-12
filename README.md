# Diabetic Retinopathy  — Baseline CNN & EfficientNet

This project uses deep learning to classify the **severity of diabetic retinopathy (DR)** from retinal fundus images.  
We provide **two implementations**:

1. **Baseline Custom CNN** — a simpler network for quick prototyping  
2. **EfficientNet-B4 (Transfer Learning)** — a high-performance model for better accuracy

Dataset: [Kaggle - Diabetic Retinopathy Resized](https://www.kaggle.com/datasets/linchundan/diabetic-retinopathy-resized)

***

***

## 📊 Dataset
- **Source:** Kaggle - *diabetic-retinopathy-resized*
- **Classes:**
  ```
  0 = No DR
  1 = Mild
  2 = Moderate
  3 = Severe
  4 = Proliferative DR
  ```

***

## ⚙️ Installation
Install dependencies:
```bash
pip install torch torchvision efficientnet_pytorch opencv-python seaborn scikit-learn imbalanced-learn
```

***

## 🧹 Preprocessing (Both Models)
1. **Remove black borders** — `crop_black()`
2. **Circular crop** — keep only the retina
3. **Gaussian blur + contrast adjustment**
4. **Augmentation** — random crop, horizontal/vertical flips
5. **Class balancing**:
   - CNN → undersample majority class (0)
   - EfficientNet → weighted loss / focal loss preferred

***

## 🏗 Model Architectures

| Feature                    | Baseline Custom CNN                          | EfficientNet-B4 (Transfer Learning)                 |
|----------------------------|-----------------------------------------------|------------------------------------------------------|
| Conv layers                | 5 Conv blocks (VGG-style)                    | Pretrained ImageNet conv layers                     |
| Params                     | ~14M                                         | ~19M                                                 |
| Classifier head            | 4096 → 4096 → 5                               | FC (1792 → 5)                                        |
| Image size                 | 256×256                                      | 380×380                                              |
| Loss                       | CrossEntropyLoss                              | CrossEntropyLoss / Focal Loss                        |
| Optimizer                  | Adam (1e-3)                                   | Adam / AdamW (3e-4)                                  |
| Scheduler                  | StepLR                                        | CosineAnnealing / ReduceLROnPlateau                  |
| Training speed             | Fast, low GPU requirements                   | Slower, needs more VRAM                              |

***

## 🚀 Training

**Baseline CNN**
```
Batch size: 16
Epochs: 2+ (example, increase for real training)
Early stopping: patience=35
```

**EfficientNet-B4**
```
Batch size: 8–16 (depends on GPU)
Epochs: 10–30
Early stopping: patience=7
```

Run notebooks:
```bash
jupyter notebook Retinopathy.ipynb
jupyter notebook Retinopathy_EfficientNet.ipynb
```

***

## 📈 Performance (Sample)

| Metric              | Baseline CNN       | EfficientNet-B4     |
|---------------------|--------------------|---------------------|
| Accuracy            | ~74%               | ~85–90%              |
| Cohen’s Kappa (val) | 0.00–0.10           | 0.75–0.85            |
| Recall (rare cls)   | Very low            | Much higher          |

***

## ▶️ Inference Example (EfficientNet)
```python
from efficientnet_pytorch import EfficientNet
import torch
from PIL import Image
import torchvision.transforms as T

# Load trained model
model = EfficientNet.from_name('efficientnet-b4')
model._fc = torch.nn.Linear(1792, 5)
model.load_state_dict(torch.load("./models/model_efficientnet_DR.bin"))
model.eval()

# Transform
transform = T.Compose([
    T.Resize((380, 380)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

# Predict
img = Image.open("sample_retina.jpg")
input_tensor = transform(img).unsqueeze(0)
with torch.no_grad():
    pred = model(input_tensor)
print("Predicted label:", torch.argmax(pred, dim=1).item())
```

***

## 📌 Summary
- **Baseline CNN** → Quick, light, but less accurate  
- **EfficientNet-B4** → More accurate, better class recall, needs GPU  

**Recommended:** Use EfficientNet-B4 with proper class balancing & 15+ epochs for strong results.

***

If you want, I can now make this README **include visual training curves & confusion matrices** for both models so it looks even more professional for GitHub/Kaggle.  

Do you want me to prepare that visual-enhanced README?
