from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import joblib
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

### Loading the dataset 
data = load_breast_cancer()
X1 = data.data
y1 = data.target

### Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=42)


### defining structure
df = pd.DataFrame(X1, columns=data.feature_names)
df['target'] = y1
feature_names = list(df.columns[:-1])



### Data Preprocessing
imputer = SimpleImputer(strategy='mean')
df[data.feature_names] = imputer.fit_transform(df[data.feature_names])
df.drop_duplicates(inplace=True)


# Feature and target separation
X = df.drop('target', axis=1)
y = df['target']


#Feature Scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

### Model Training
model = SVC(kernel='rbf', random_state=42, probability=True)
model.fit(x_train, y_train)

### Model Prediction
y_pred = model.predict(x_test)

### Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

### saving the model
joblib.dump(model, "cancer_model.pkl")
joblib.dump(scaler, "scaler.pkl")


# ================= GUI START =====================

root = ttk.Window(themename="cyborg")
root.title("Cancer Prediction System")
root.geometry("1400x900")

# Notebook container
notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True, pady=10, padx=10)


# ---------- Prediction Tab ----------
predict_page = ttk.Frame(notebook, padding=20)
notebook.add(predict_page, text="Prediction")

title = ttk.Label(
    predict_page,
    text="Breast Cancer Prediction Dashboard",
    font=("Helvetica", 26, "bold"),
    bootstyle="primary"
)
title.pack(pady=15)

form_frame = ttk.Frame(predict_page)
form_frame.pack(pady=20)

input_boxes = []

for i, feature in enumerate(feature_names):
    r = i // 3
    c = (i % 3) * 2

    ttk.Label(
        form_frame,
        text=feature,
        font=("Helvetica", 11)
    ).grid(row=r, column=c, sticky="w", padx=8, pady=6)

    box = ttk.Entry(
        form_frame,
        width=12,
        font=("Helvetica", 11)
    )
    box.grid(row=r, column=c+1, padx=8, pady=6)

    try:
        box.insert(0, round(float(df[feature].mean()), 3))
    except:
        box.insert(0, 0)

    input_boxes.append(box)

result_label = ttk.Label(
    predict_page,
    text="Prediction will appear here...",
    font=("Helvetica", 20, "bold"),
    bootstyle="info"
)
result_label.pack(pady=35)

def gui_predict():
    try:
        values = [float(x.get()) for x in input_boxes]
        arr = np.array(values).reshape(1, -1)
        arr = scaler.transform(arr)

        pred = model.predict(arr)[0]

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(arr)[0][1]
        else:
            score = model.decision_function(arr)[0]
            prob  = 1 / (1 + np.exp(-score))

        if pred == 1:
            result_label.config(text=f"Malignant Tumor\nProbability: {prob*100:.2f}%", bootstyle="danger")
        else:
            result_label.config(text=f"Benign Tumor\nConfidence: {(1-prob)*100:.2f}%", bootstyle="success")

    except Exception as e:
        result_label.config(text=f"Error → {e}", bootstyle="warning")


predict_btn = ttk.Button(
    predict_page,
    text="PREDICT",
    bootstyle=SUCCESS,
    width=30,
    padding=15,
    command=gui_predict
)
predict_btn.pack(pady=10)


# ---------- Visualization Tab ----------
visual_page = ttk.Frame(notebook, padding=20)
notebook.add(visual_page, text="Visualization")

ttk.Label(
    visual_page,
    text="Feature Distribution Visualizer",
    font=("Helvetica", 22, "bold")
).pack(pady=10)

combo = ttk.Combobox(visual_page, values=list(feature_names), width=40, font=("Helvetica", 12))
combo.set(feature_names[0])
combo.pack(pady=10)

plot_frame = ttk.Frame(visual_page)
plot_frame.pack(fill="both", expand=True, padx=20, pady=20)

def show_plot():
    feature = combo.get()
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df[feature], kde=True, ax=ax)
    ax.set_title(f"Distribution: {feature}")

    for w in plot_frame.winfo_children():
        w.destroy()

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

ttk.Button(
    visual_page,
    text="Generate Plot",
    bootstyle=PRIMARY,
    command=show_plot
).pack(pady=5)


# ---------- Model Metrics Tab ----------
metrics_page = ttk.Frame(notebook, padding=20)
notebook.add(metrics_page, text="Model Metrics")

ttk.Label(
    metrics_page,
    text="Model Evaluation Metrics",
    font=("Helvetica", 24, "bold"),
    bootstyle="primary"
).pack(pady=10)

import sklearn.metrics as mt

y_true = y_test
y_pred = model.predict(x_test)

acc  = mt.accuracy_score(y_true, y_pred)
prec = mt.precision_score(y_true, y_pred)
rec  = mt.recall_score(y_true, y_pred)
f1   = mt.f1_score(y_true, y_pred)

metrics_text = f"""
Accuracy:  {acc*100:.2f}%
Precision: {prec*100:.2f}%
Recall:    {rec*100:.2f}%
F1 Score:  {f1*100:.2f}%
"""


ttk.Label(
    metrics_page,
    text=metrics_text.strip(),
    font=("Helvetica", 16),
    bootstyle="info"
).pack(pady=10)

cm_frame = ttk.Frame(metrics_page)
cm_frame.pack(fill="both", expand=True, padx=20, pady=20)

def show_confusion():
    for w in cm_frame.winfo_children():
        w.destroy()

    cm = mt.confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="magma", ax=ax)
    ax.set_title("Confusion Matrix")

    canvas = FigureCanvasTkAgg(fig, master=cm_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

ttk.Button(
    metrics_page,
    text="Show Confusion Matrix",
    bootstyle=PRIMARY,
    command=show_confusion
).pack(pady=10)

roc_frame = ttk.Frame(metrics_page)
roc_frame.pack(fill="both", expand=True, padx=20, pady=20)

def show_roc():
    for w in roc_frame.winfo_children():
        w.destroy()

    try:
        y_prob = model.predict_proba(x_test)[:, 1]
    except:
        y_prob = model.decision_function(x_test)

    fpr, tpr, _ = mt.roc_curve(y_true, y_prob)
    auc = mt.roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(4,4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0,1],[0,1],'--')
    ax.set_title("ROC Curve")
    ax.legend()

    canvas = FigureCanvasTkAgg(fig, master=roc_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

ttk.Button(
    metrics_page,
    text="Show ROC Curve",
    bootstyle=SECONDARY,
    command=show_roc
).pack(pady=10)

root.mainloop()
