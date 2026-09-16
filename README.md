# PashuRaksha AI — Livestock Health Screening Pipeline

An end-to-end, reproducible AI/ML pipeline for **livestock disease screening** that combines:

- **MobileNetV3** image classifier (transfer learning, two-stage fine-tuning)
- **XGBoost** clinical risk model (auto-detects binary vs. multiclass)
- **Isolation Forest** anomaly / unknown-pattern detector
- Transparent **weighted multimodal fusion** engine
- **Grad-CAM** image explainability
- **SHAP** clinical explainability
- Optional **DBSCAN** outbreak / cluster analytics
- End-to-end pipeline orchestrator with reproducibility built in

> **Medical disclaimer.** PashuRaksha AI is an AI **screening** prototype — it is
> **not** a confirmed veterinary diagnosis. Whenever confidence is low, the
> pattern is unknown, symptoms are severe, or laboratory confirmation is normally
> required, the output explicitly recommends veterinary examination and/or
> laboratory confirmation.

---

## 1. Project Layout

```
pashuraksha-ai/
├── data/
│   ├── raw/
│   ├── images/             # place image dataset here as images/<class>/<file>.jpg
│   ├── clinical/           # place clinical CSV/XLSX here
│   └── processed/
├── data_processing/
│   ├── inspect_dataset.py     # spec §3, §15
│   ├── clean_images.py        # spec §4
│   ├── prepare_images.py      # spec §5, §6
│   └── prepare_clinical.py    # spec §16
├── training/
│   ├── train_image.py         # spec §7-§10, §12   (MobileNetV3 + temperature scaling)
│   ├── train_clinical.py      # spec §17-§19       (XGBoost)
│   ├── train_anomaly.py       # spec §21-§23       (Isolation Forest on embeddings)
│   ├── evaluate_image.py     # spec §11
│   └── evaluate_clinical.py   # spec §19
├── inference/
│   ├── image_predict.py       # spec §13, §14
│   ├── clinical_predict.py   # spec §20
│   └── multimodal_predict.py  # spec §24-§33, §52
├── explainability/
│   ├── image_explain.py       # spec §29   (Grad-CAM)
│   └── clinical_explain.py    # spec §30   (SHAP)
├── analytics/
│   └── disease_clusters.py    # spec §34, §35  (DBSCAN + temporal trends)
├── models/
│   ├── image_model/            # best_model.pth saved here
│   ├── clinical_model/         # model.joblib, preprocessor.joblib, etc.
│   └── anomaly_model/          # image_anomaly.joblib, clinical_anomaly.joblib
├── reports/
│   ├── image_training/         # training curves, history JSON
│   ├── clinical_training/      # training config JSON
│   ├── explanations/{images,clinical}/
│   ├── dataset_report.json
│   ├── dataset_split.csv
│   ├── image_test_metrics.json
│   ├── clinical_metrics.json
│   ├── confusion_matrix.png
│   └── final_model_report.md
├── configs/
│   └── default.yaml            # central configuration
├── runs/                       # experiment outputs
├── notebooks/
├── utils/                     # seed, device, metrics, logger
├── requirements.txt
├── README.md
├── .gitignore
└── run_pipeline.py             # full 15-step orchestrator (spec §57)
```

---

## 2. Installation

```bash
python -m venv .venv
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

**Windows:**
```powershell
.venv\Scripts\activate
```

Then:

```bash
pip install -r requirements.txt
```

Tested with Python 3.11+.

---

## 3. Reproducibility

A single global seed drives Python `random`, NumPy, and PyTorch (including CUDA
and cuDNN determinism). The default seed is `42` and is configurable in
`configs/default.yaml` and via the `--seed` flag on every CLI script.

Every saved checkpoint records:
- `class_names`, `class_to_idx`
- `model_name`, `input_size`, `normalization`
- `training_config`
- `best_validation_metric`
- `temperature` (calibration)
- reproducibility info (Python version + package versions)

---

## 4. Step-by-Step Execution

### 4.1 Run the entire pipeline

```bash
python run_pipeline.py --epochs 30 --batch-size 32 --seed 42
```

This runs all 15 steps from spec §57 in the correct order:

1. Discover datasets
2. Inspect datasets
3. Report dataset statistics
4. Clean invalid / duplicate records
5. Create stratified 70 / 15 / 15 split
6. Train MobileNetV3 image classifier
7. Evaluate image classifier
8. Train clinical XGBoost model
9. Evaluate clinical model
10. Extract MobileNetV3 embeddings
11. Train anomaly detector
12. Build multimodal fusion engine
13. Add explainability (Grad-CAM + SHAP)
14. Generate final model evaluation report
15. Demonstrate inference using real samples

If no dataset is found, the pipeline implements itself end-to-end and clearly
reports `Pipeline implemented` (NOT `Model trained and validated`) per spec §58.

### 4.2 Run individual stages

```bash
# Inspect only
python data_processing/inspect_dataset.py --all

# Image dataset
python data_processing/clean_images.py --data-dir data/images
python data_processing/prepare_images.py --data-dir data/images
python training/train_image.py --data-dir data/images --epochs 50 --batch-size 32
python training/evaluate_image.py --model models/image_model/best_model.pth

# Clinical dataset
python data_processing/prepare_clinical.py --data data/clinical/clinical.csv
python training/train_clinical.py --data data/clinical/clinical.csv
python training/evaluate_clinical.py --data data/clinical/clinical.csv

# Anomaly detector
python training/train_anomaly.py \
    --image-model models/image_model/best_model.pth \
    --data-dir data/images

# Inference
python inference/image_predict.py --image test.jpg
python inference/clinical_predict.py --age 4 --temperature 40.2 --symptoms "fever,loss_of_appetite"
python inference/multimodal_predict.py \
    --image cow.jpg \
    --age 4 \
    --breed "Gir" \
    --temperature 40.2 \
    --symptoms "fever,loss_of_appetite"

# Explainability
python explainability/image_explain.py --image sample.jpg
python explainability/clinical_explain.py --data data/clinical/clinical.csv --row-index 0

# Analytics (only when geo+date columns exist)
python analytics/disease_clusters.py --data data/clinical/clinical.csv
```

---

## 5. Configuration

All configurable values live in `configs/default.yaml`. Highlights:

| Section        | Key setting                                            |
|----------------|--------------------------------------------------------|
| `reproducibility` | `random_seed: 42`                                  |
| `image`        | `input_size: 224`, ImageNet normalization              |
| `image.split`  | 70 / 15 / 15                                            |
| `image_model`  | `mobilenet_v3_large`, stage1 + stage2 training         |
| `image_model.early_stopping` | patience 6, metric macro_f1              |
| `image_model.calibration`   | temperature scaling                   |
| `image_model.confidence_threshold` | `0.55` (prototype decision threshold) |
| `clinical_model` | XGBoost, auto binary/multiclass detection            |
| `anomaly_model` | Isolation Forest on MobileNetV3 embeddings            |
| `fusion`       | transparent weighted fusion + disagreement threshold   |
| `fusion.safety` | extreme temperature / severe symptom triggers         |
| `risk.bands`   | 0–30 Low / 31–60 Moderate / 61–80 High / 81–100 Critical |

---

## 6. Safety Guarantees

Per spec §54, the pipeline **always** preserves the distinction:

```
AI Screening  ≠  Confirmed Diagnosis
```

Output language uses:

- "Possible indication"
- "Screening prediction"
- "Pattern associated with"
- "Estimated health risk"
- "Veterinary confirmation recommended"
- "Laboratory confirmation may be required"

The pipeline **never** produces "The animal definitely has Disease X."

### Safety overrides (spec §28)

When severe observations appear in the input (extreme temperature, severe
symptom keywords such as `inability_to_stand`, `severe_breathing_difficulty`,
`seizures`, `severe_dehydration`, `uncontrolled_bleeding`), the system forces
the risk level into at least `High` and returns "Urgent veterinary assessment
recommended." Symptoms are never fabricated.

### Unknown pattern handling (spec §22, §55)

When either:
- `anomaly_score > threshold`, OR
- `max known-class confidence < confidence_threshold`

the system returns `Unknown / Unusual Pattern` with:

> AI could not reliably match this case to known patterns.
> Veterinary examination and, where appropriate, laboratory testing are recommended.

---

## 7. Data Leakage Protection (spec §56)

- Stratified splits with deterministic seed
- Duplicate image detection (MD5 hashing) — duplicates never appear across
  train / val / test
- Preprocessing (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`,
  `MultiSymptomEncoder`) is fit **only** on training data
- Validation and test data remain untouched until their respective evaluation
  stages

---

## 8. Final Deliverables

After running the pipeline you should have:

1. Complete AI/ML source code under `data_processing/`, `training/`,
   `inference/`, `explainability/`, `analytics/`, `utils/`
2. Dataset inspection report at `reports/dataset_report.json`
3. Dataset cleaning reports at `reports/corrupted_images.csv`,
   `reports/duplicate_images.csv`, `reports/dataset_report.json`
4. Train/val/test split at `reports/dataset_split.csv`
5. MobileNetV3 image model at `models/image_model/best_model.pth`
6. XGBoost clinical model at `models/clinical_model/model.joblib`
7. Anomaly detection model at `models/anomaly_model/image_anomaly.joblib`
8. Multimodal fusion engine (in `inference/multimodal_predict.py`)
9. Grad-CAM image explainability heatmaps in `reports/explanations/images/`
10. SHAP clinical explainability artifacts in `reports/explanations/clinical/`
11. Training graphs in `reports/image_training/` (loss / accuracy / f1 / lr curves)
12. Confusion matrices in `reports/confusion_matrix.png` and
    `reports/clinical_confusion_matrix.png`
13. Per-class evaluation in `reports/image_classification_report.csv`
14. Saved deployable model files under `models/`
15. `requirements.txt`
16. This README
17. Reproducible training commands (see Section 4)
18. Final model performance report at `reports/final_model_report.md`
19. Example inference output (printed to stdout by Step 15)

---

## 9. Honest Status Reporting (spec §58)

The pipeline **never** fabricates:

- datasets, dataset sizes, disease classes, samples per class
- accuracy, precision, recall, F1, ROC-AUC
- confusion matrices, validation results, disease probabilities, training graphs

When the required dataset is unavailable, the final report explicitly states:

> Training cannot be completed because the required dataset is not available.
>
> The pipeline has been implemented and is runnable once real data is placed
> under `data/images/<class>/` or `data/clinical/<file>.csv`.

The pipeline clearly distinguishes **Pipeline implemented** from **Model trained
and validated**.

---

## 10. Hackathon Demonstration Flow

```
Livestock Image
       │
       ▼
MobileNetV3
       │
       ├── Disease probabilities
       └── Image embedding
               │
               ▼
        Anomaly Detector
               │
               ▼
        Unknown Pattern Check

Clinical Information
       │
       ▼
Clinical Preprocessing
       │
       ▼
XGBoost
       │
       ▼
Clinical Probabilities

Image Evidence
       +
Clinical Evidence
       +
Health History
       +
Anomaly Status
       │
       ▼
Multimodal Fusion
       │
       ▼
Disease Screening Prediction
       +
Risk Score 0–100
       +
Risk Level
       +
Contributing Factors
       +
Veterinary/Lab Recommendation
```

To demonstrate end-to-end with a real sample:

```bash
python inference/multimodal_predict.py \
    --image sample.jpg \
    --age 4 --breed "Gir" --temperature 40.2 \
    --symptoms "fever,loss_of_appetite"
```

---

## 11. License & Disclaimer

This prototype is provided for research and demonstration purposes only. It is
not a substitute for professional veterinary advice, diagnosis, or treatment.
Always seek the advice of a qualified veterinarian with any questions about a
medical condition.
# datasete-2
