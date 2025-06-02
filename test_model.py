import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from tensorflow.keras.models import load_model
import os

# Create directory for saving plots
os.makedirs('model_metrics_testing', exist_ok=True)

# Load test dataset
ds = r'C:\Users\Isuru\Downloads\finalone\junction_analysistest1.csv'
df = pd.read_csv(ds)

# Convert Label column to numeric
def convert_labels(labels):
    # Handle common non-numeric formats
    label_map = {
        'green': 1, 'red': 0,
    }
    try:
        # Map known labels
        numeric_labels = labels.map(label_map)
        if numeric_labels.isna().any():
            # If there are unmapped labels, print unique values for debugging
            unmapped = labels[numeric_labels.isna()].unique()
            raise ValueError(f"Unrecognized label values: {unmapped}. Please ensure all labels are in {list(label_map.keys())}.")
        return numeric_labels.astype(int)
    except Exception as e:
        raise ValueError(f"Error converting labels: {str(e)}")

df['Label'] = convert_labels(df['Label'])

# Prepare data
features = [
    'People_Crossing', 'People_In_Crosswalk_Now', 'Avg_People_Velocity_Crossing_kmph',
    'People_Waiting', 'People_In_Waiting_Now', 'Avg_People_Velocity_Waiting_kmph',
    'Vehicles', 'Avg_Vehicle_Velocity_kmph', 'Queued_Vehicles'
]
X = df[features].values
y = df['Label'].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply feature weights
# Indices of features list
weights = np.ones(len(features))  # Default weight of 1 for all features
weights[0] = 1.0 # Higher weight for People_Crossing
weights[1] = 1.0 # Higher weight for People_In_Crosswalk_Now
weights[2] = 1.0 # Higher weight for Avg_People_Velocity_Crossing_kmph
weights[3] = 1.0 # Higher weight for People_Waiting
weights[4] = 8.0 # Higher weight for People_In_Waiting_Now
weights[5] = 1.0 # Higher weight for Avg_People_Velocity_Waiting_kmph
weights[6] = 1.0 # Higher weight for Vehicles
weights[7] = 1.0  # Higher weight for Avg_Vehicle_Velocity_kmph
weights[8] = 1.0  # Higher weight for Queued_Vehicles
X_weighted = X_scaled * weights  # Apply weights to standardized features

# Create sequences for LSTM
sequence_length = 8  # Match training sequence length
X_seq, y_seq = [], []
for i in range(len(X_weighted) - sequence_length):
    X_seq.append(X_weighted[i:i + sequence_length])
    y_seq.append(y[i + sequence_length])
X_seq = np.array(X_seq)  # Shape: (samples, sequence_length, n_features)
y_seq = np.array(y_seq)  # Shape: (samples,)

# Load the saved model
model = load_model('lstm_traffic_model.h5')

# Evaluate model
y_pred = (model.predict(X_seq) > 0.5).astype(int)

# Calculate and print metrics
precision = precision_score(y_seq, y_pred)
recall = recall_score(y_seq, y_pred)
f1 = f1_score(y_seq, y_pred)
accuracy = accuracy_score(y_seq, y_pred)

print("\nTest Dataset Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Confusion matrix
cm = confusion_matrix(y_seq, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Test Dataset)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('model_metrics_testing/confusion_matrix_test.png')
plt.close()

# ROC curve
y_pred_proba = model.predict(X_seq)
fpr, tpr, _ = roc_curve(y_seq, y_pred_proba)
auc = roc_auc_score(y_seq, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve (Test Dataset)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig('model_metrics_testing/roc_curve_test.png')
plt.close()

print("\nEvaluation complete. Plots saved in 'model_metrics_testing' folder.")