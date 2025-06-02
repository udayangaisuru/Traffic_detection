import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

# Create directories for saving plots
os.makedirs('correlation', exist_ok=True)
os.makedirs('model_metrics_testing', exist_ok=True)

# Load dataset
ds = r'C:\Users\Isuru\Downloads\finalone\junction_analysis1.csv'
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

# Add encoded columns L (red) and M (green)
# Assuming L=1 when Label=0 (red), M=1 when Label=1 (green)
df['green'] = (df['Label'] == 0).astype(int)
df['red'] = (df['Label'] == 1).astype(int)

# Generate and save correlation matrix (including L and M)
plt.figure(figsize=(12, 12))
corr = df.drop(columns=['Timestamp']).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Traffic Dataset (with Red and Green Labels)')
plt.savefig('correlation/correlation_matrix.png')
plt.close()

# Prepare data (exclude L and M from features)
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
# Indices of 'Avg_Vehicle_Velocity_kmph' (7) and 'Queued_Vehicles' (8) in features list
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
sequence_length = 8
X_seq, y_seq = [], []
for i in range(len(X_weighted) - sequence_length):
    X_seq.append(X_weighted[i:i + sequence_length])
    y_seq.append(y[i + sequence_length])
X_seq = np.array(X_seq)  # Shape: (samples, sequence_length, n_features)
y_seq = np.array(y_seq)  # Shape: (samples,)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential([
    LSTM(64, input_shape=(sequence_length, len(features)), return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# Plot and save training history
plt.figure(figsize=(12, 4))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('model_metrics_testing/loss_plot.png')
plt.close()

# Accuracy plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('model_metrics_testing/accuracy_plot.png')
plt.close()

# Evaluate model
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Calculate and print metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("\nTest Set Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('model_metrics_testing/confusion_matrix.png')
plt.close()

# ROC curve
y_pred_proba = model.predict(X_test)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig('model_metrics_testing/roc_curve.png')
plt.close()

model.save('lstm_traffic_model.h5')

print("\nModel training complete. Plots saved in 'correlation' and 'model_metrics_testing' folders.")