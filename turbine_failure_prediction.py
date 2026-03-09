"""
ReneWind - Wind Turbine Failure Prediction

Objective:
Build machine learning classification models to predict wind turbine generator failures
in order to reduce maintenance cost.

Target:
1 = Failure
0 = No Failure
"""

# ===============================
# Import Required Libraries
# ===============================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.utils import class_weight

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

import time
import random
import warnings

warnings.filterwarnings("ignore")


# ===============================
# Reproducibility
# ===============================

def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


set_seeds()


# ===============================
# Load Training Data
# ===============================

TRAIN_PATH = "data/Train.csv"

renewind_train = pd.read_csv(TRAIN_PATH)

# Copy dataset
df = renewind_train.copy()


# ===============================
# Exploratory Data Analysis
# ===============================

print("Shape:", df.shape)
print(df.info())
print(df.describe())

print("Duplicate rows:", df.duplicated().sum())

print("Target distribution:")
print(df["Target"].value_counts(normalize=True))

print("Missing values:")
print(df.isnull().sum())


# ===============================
# Visualization
# ===============================

numeric_cols = df.select_dtypes(include="number").columns.drop("Target")

# Histogram
df[numeric_cols].hist(figsize=(12, 10), bins=30)
plt.suptitle("Histograms of Numeric Features")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df[numeric_cols].corr(), cmap="YlGnBu")
plt.title("Feature Correlation Heatmap")
plt.show()


# ===============================
# Data Preprocessing
# ===============================

X = df.drop(columns=["Target"])
y = df["Target"]

# Train validation split
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

# Impute missing values
imputer = SimpleImputer(strategy="median")

imputer.fit(X_train[["V1", "V2"]])

X_train[["V1", "V2"]] = imputer.transform(X_train[["V1", "V2"]])
X_val[["V1", "V2"]] = imputer.transform(X_val[["V1", "V2"]])

# Scale features
scaler = StandardScaler()

scaler.fit(X_train)

X_train = pd.DataFrame(
    scaler.transform(X_train),
    columns=X_train.columns,
    index=X_train.index,
)

X_val = pd.DataFrame(
    scaler.transform(X_val),
    columns=X_val.columns,
    index=X_val.index,
)


# ===============================
# Model Utility Functions
# ===============================

def plot(history, metric):

    plt.figure()

    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])

    plt.title("Model " + metric.capitalize())
    plt.ylabel(metric.capitalize())
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"])

    plt.show()


def model_performance_classification(model, X, y_true):

    y_pred_probs = model.predict(X)

    y_pred = (y_pred_probs > 0.5).astype("int32")

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred),
    }


# ===============================
# Training Parameters
# ===============================

EPOCHS = 25
BATCH_SIZE = 64


# ===============================
# Compute Class Weights
# ===============================

weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train,
)

class_weights = {0: weights[0], 1: weights[1]}


# ===============================
# Final Model Architecture
# ===============================

tf.keras.backend.clear_session()

model = Sequential()

model.add(Dense(64, activation="relu", input_dim=X_train.shape[1]))
model.add(Dropout(0.2))

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(1, activation="sigmoid"))

model.summary()

model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["Recall"],
)


# ===============================
# Train Model
# ===============================

start = time.time()

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    class_weight=class_weights,
)

end = time.time()

print("Training Time:", end - start)

plot(history, "loss")


# ===============================
# Validation Performance
# ===============================

train_perf = model_performance_classification(model, X_train, y_train)
val_perf = model_performance_classification(model, X_val, y_val)

print("Training Performance:", train_perf)
print("Validation Performance:", val_perf)


# ===============================
# Confusion Matrix
# ===============================

y_val_pred_prob = model.predict(X_val)

y_val_pred = (y_val_pred_prob >= 0.5).astype(int)

cm = confusion_matrix(y_val, y_val_pred)

disp = ConfusionMatrixDisplay(cm)

disp.plot(cmap=plt.cm.Blues)

plt.title("Validation Confusion Matrix")

plt.show()


# ===============================
# Test Dataset Evaluation
# ===============================

TEST_PATH = "data/Test.csv"

renewind_test = pd.read_csv(TEST_PATH)

test_df = renewind_test.copy()

X_test = test_df.drop(columns=["Target"])
y_test = test_df["Target"]

X_test[["V1", "V2"]] = imputer.transform(X_test[["V1", "V2"]])

X_test = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index,
)

y_test_pred_prob = model.predict(X_test)

y_test_pred = (y_test_pred_prob >= 0.5).astype(int)

print("Test Classification Report")

print(classification_report(y_test, y_test_pred))


cm_test = confusion_matrix(y_test, y_test_pred)

disp = ConfusionMatrixDisplay(cm_test)

disp.plot(cmap=plt.cm.Blues)

plt.title("Test Confusion Matrix")

plt.show()
