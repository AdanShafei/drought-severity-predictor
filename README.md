# Drought Severity Predictor — Neural Network & XGBoost

This project predicts **drought severity levels (D0–D3)** in California using key environmental indicators: **temperature**, **precipitation**, **soil moisture**, and **historical drought data**.

We implement and compare a **Neural Network (Keras/TensorFlow)** and **XGBoost**.  
A small user-input step in the notebook demonstrates how raw values (precipitation, temperature, soil moisture) map to a predicted drought class.

Originally developed by **Group 20 (CSE 144)**, this repository is cleaned and organized for portfolio and academic purposes.

---

## 🌎 Background

California experiences recurring drought cycles with substantial ecological and socioeconomic impact.  
This work demonstrates how classical ML and deep learning models can classify weekly drought levels from climate variables, offering a data-driven baseline for environmental decision support.

---

## 🧠 Methodology

### **1. Data Collection**
All datasets were obtained from open environmental sources:
- **Historical drought data:** U.S. Drought Monitor  
- **Precipitation:** California Department of Water Resources  
- **Soil moisture:** U.S. Geological Survey (USGS)  
- **Temperature:** NOAA / California-Nevada River Forecast Center  

Each dataset was preprocessed and aligned by time frame to ensure consistency.

---

### **2. Data Processing**
- Converted all data to numeric form and imputed missing values with zeros.  
- Concatenated all features into a single dataset (`X`) and created derived features:
  - Precipitation-to-temperature ratio  
  - Temporal differences (e.g., precipitation_diff, soil_moisture_diff)  
  - Temperature squared (for nonlinear effects)
- Target variable (`y`) categorized drought levels (D0–D3) using `pandas.cut`.
- Addressed class imbalance using **SMOTETomek** (oversampling + Tomek link removal).
- Standardized features with `StandardScaler` for model stability.

---

## 🤖 Model Development

### **Neural Network (Keras)**
- Fully Connected Feedforward Network  
- ReLU activation, Batch Normalization, Dropout  
- Early Stopping to prevent overfitting  
- Achieved **~76–77% test accuracy** and **85.25% balanced accuracy**

### **XGBoost Classifier**
- Gradient-boosted trees with L1/L2 regularization  
- Tuned hyperparameters for optimal recall and precision  
- Achieved **~80–82% validation accuracy**

> Combining XGBoost and the Neural Network improved robustness and interpretability.  
> XGBoost captured feature importance while the NN modeled complex nonlinear relationships.

---

## 📊 Results

| Model | Train Accuracy | Validation/Test Accuracy | Balanced Accuracy |
|-------|----------------|--------------------------|-------------------|
| Neural Network | 82% | 76–77% | **85.25%** |
| XGBoost | **85%** | **~82%** | 80% |
| Ensemble | 83% | 80% | — |

**Feature Importance Highlights:**
- 🌡️ **Temperature** and **Soil Moisture** were the strongest predictors  
- 🌧️ **Precipitation** and its ratio to temperature added valuable variation  

The models correctly classified the extremes (D0 and D3) most confidently,  
while D1 and D2 levels showed expected ambiguity due to dataset imbalance.

---

## 📈 Example Outputs

- **Accuracy Curves:** XGBoost stable around 80%, NN peaks near 77% after 50 epochs  
- **User Input Demo:** Given (precipitation=80, temperature=90, soil moisture=0.6), both models predicted **D3 (Severe Drought)**  
- **Confusion Matrix:** High precision for D0 & D3; improvement needed for D1–D2

---

## ⚠️ Limitations
- **Class imbalance:** Few samples for D1–D2 affected midrange accuracy.  
- **Data availability:** Some datasets were incomplete or inconsistently scaled.  
- **Scope:** Focused only on 3–4 environmental variables; real droughts depend on more complex patterns.

---

## 🚀 Future Work
- Include **ENSO indices**, **vegetation data**, and **socioeconomic indicators**.  
- Expand training data beyond 15 years to balance D1–D2 occurrences.  
- Add a **front-end dashboard** with map-based drought visualization.  
- Explore deeper neural networks and ensemble stacking.

---

## 📂 Repository Structure

<pre><code>.
├── notebooks/
│   └── drought_severity_predictor.ipynb
├── data/
│   ├── historical_drought/
│   ├── precipitation/
│   ├── soil-moisture/
│   └── temperature/
├── Drought Project Report.pdf
└── README.md
</code></pre>

---

## ▶️ How to Run

To explore or reproduce the results:
```bash
jupyter notebook notebooks/drought_severity_predictor.ipynb
