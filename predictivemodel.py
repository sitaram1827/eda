import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

def run_predictive_pipeline(csv_path):
    print("--- 1. LOADING AND PREPARING DATA ---")
    if not os.path.exists(csv_path):
        print(f"Error: The file '{csv_path}' was not found. Please create it first.")
        return
        
    df = pd.read_csv(csv_path)

    # Drop rows with missing values and ensure target is numeric
    df = df.dropna()
    if df.shape[0] < 2:
        print("Error: Not enough data after dropping missing values.")
        return

    # Separate the target class from the input features (last column)
    df.iloc[:, -1] = pd.to_numeric(df.iloc[:, -1], errors='coerce')
    df = df.dropna()
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].astype(int)

    # Need at least two classes to train a classifier
    if y.nunique() < 2:
        print("Error: target column needs at least two classes for training.")
        return
    
    # Split data: 70% Training, 30% Testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"Dataset successfully loaded. Total rows: {len(df)}")
    print(f"Training set rows: {len(X_train)} | Testing set rows: {len(X_test)}\n")

    print("--- 2. TRAINING MODEL (RANDOM FOREST) ---")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model training complete.\n")

    print("--- 3. TERMINAL OUTPUT: EVALUATION METRICS ---")
    y_pred = model.predict(X_test)

    # compute probabilities only if available and binary
    y_probs = None
    if hasattr(model, 'predict_proba') and len(model.classes_) == 2:
        y_probs = model.predict_proba(X_test)[:, 1]

    print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("--- 4. DISPLAYING PERFORMANCE PLOTS ---")

    # close any leftover figures
    plt.close('all')

    # Confusion matrix (show, not save)
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
    disp.plot(ax=ax, cmap='Blues')   # pass ax to avoid creating a second figure
    ax.set_title('Confusion Matrix')
    plt.show()

    # ROC (only when probabilities available)
    if y_probs is not None:
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc='lower right')
        ax.grid(True)
        plt.show()

    print("\nPipeline executed successfully!")

if __name__ == "__main__":
    # Specify your local input CSV file path here
    run_predictive_pipeline('data_set.csv')
