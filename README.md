# Turbine Failure Prediction

Deep learning model to predict wind turbine generator failures from sensor data, enabling proactive maintenance and reduced operational downtime.

---

## Problem

Unexpected turbine failures can cause costly repairs and operational downtime. Predicting failures early allows maintenance teams to perform preventive maintenance and avoid expensive generator replacements.

---

## Approach

- Neural network built using TensorFlow/Keras
- Data preprocessing with missing value imputation and feature scaling
- Class imbalance handled using class weights
- Model evaluation focused on recall to minimize missed failures

---

## Results

- Failure-class recall improved from **32% → ~92%**
- Test recall for failure class: **~88%**
- Significant reduction in false negatives

---

## Tech Stack

- Python  
- Deep Learning (TensorFlow / Keras)  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  
- Seaborn  

---

## Project Structure

```text
turbine-failure-prediction/
│
├── turbine_failure_prediction.py
├── requirements.txt
├── README.md
└── data/
```


---

## Dataset

The dataset is not included in the repository.

Before running the project, place the following files inside the `data/` folder:
```
data/Train.csv
data/Test.csv

```


---

## Setup

### Clone the repository

```bash
git clone https://github.com/your username/turbine-failure-prediction.git
cd turbine-failure-prediction
Install dependencies
pip install -r requirements.txt
Run the model
python turbine_failure_prediction.py
```
Key Insight

The model prioritizes high recall to minimize missed turbine failures.
Early failure detection helps reduce downtime, optimize maintenance scheduling, and lower operational costs.
