import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import xgboost as xgb
from sklearn.metrics import ConfusionMatrixDisplay


# Ryan McDonald
# IMGS-589 ML for Remote Sensing
# Homework 3

# Problem 1
# a

data = loadmat("PaviaU.mat")
ground_truth = loadmat("PaviaU_gt.mat")

array = np.array(data['paviaU'])

def correlation_matrix(data):

    bands_flattened = data.reshape(-1, data.shape[-1])
    corr_matrix = np.corrcoef(bands_flattened, rowvar=False)

    return corr_matrix

corr_matrix = correlation_matrix(array)

plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, cmap='coolwarm', vmin=0, vmax=1)
plt.colorbar(label='Correlation Coefficient')
plt.title("Correlation PaviaU data")
plt.xlabel("Bands")
plt.ylabel("Bands")
plt.show()

pavia = loadmat("PaviaU.mat")['paviaU']
pavia_gt = loadmat("PaviaU_gt.mat")['paviaU_gt']

band_indices = [4, 24, 44, 58, 84]  # Blue, Green, Red, Red-Edge, NIR

pseudocolor = np.dstack((pavia[:, :, 84], pavia[:, :, 44], pavia[:, :, 24]))

plt.figure(figsize=(6, 6))
plt.imshow(pseudocolor / pseudocolor.max())
plt.title("Pseudocolor Image (Green, Red, NIR)")
plt.show()

# --------------------------------------------------------------------------
# b
band_indices = [4, 24, 44, 58, 84]  # Blue, Green, Red, Red-Edge, NIR

selected_bands = pavia[:, :, band_indices]
print(f"Selected Bands Shape: {selected_bands.shape}")

vegetation_classes = {2, 4}
binary_labels = np.isin(pavia_gt, list(vegetation_classes)).astype(int)

unique, counts = np.unique(binary_labels, return_counts=True)
class_distribution = dict(zip(unique, counts))

print(f"Class distribution: {class_distribution}")

X = selected_bands.reshape(-1, 5)  # Features
y = binary_labels.flatten()        # Labels

print(f"Feature Matrix Shape: {X.shape}")
print(f"Label Vector Shape: {y.shape}")

mask = pavia_gt != 0  # Keep only labeled pixels
X_filtered = selected_bands[mask]
y_filtered = binary_labels[mask]

# Reshape data into 2D for classification
X_final = X_filtered.reshape(-1, 5)
y_final = y_filtered.reshape(-1)

print(f"Filtered dataset size: {X_final.shape[0]}")

# Splitting Training and test
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, stratify=y_final, random_state=42)

# Check class distribution again
train_unique, train_counts = np.unique(y_train, return_counts=True)
test_unique, test_counts = np.unique(y_test, return_counts=True)

print(f"Corrected Training Class Distribution: {dict(zip(train_unique, train_counts))}")
print(f"Corrected Testing Class Distribution: {dict(zip(test_unique, test_counts))}")

#Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features standardized check")

clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
clf.fit(X_train_scaled, y_train)

y_train_pred = clf.predict(X_train_scaled)
y_test_pred = clf.predict(X_test_scaled)

# Get probabilities for ROC curve
y_test_probs = clf.predict_proba(X_test_scaled)[:, 1]


# Compute Metrics
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

train_precision = precision_score(y_train, y_train_pred)
test_precision = precision_score(y_test, y_test_pred)

train_recall = recall_score(y_train, y_train_pred)
test_recall = recall_score(y_test, y_test_pred)

train_f1 = f1_score(y_train, y_train_pred)
test_f1 = f1_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Testing Accuracy: {test_acc:.4f}")
print(f"Training Precision: {train_precision:.4f}")
print(f"Testing Precision: {test_precision:.4f}")
print(f"Training Recall: {train_recall:.4f}")
print(f"Testing Recall: {test_recall:.4f}")
print(f"Training F1-score: {train_f1:.4f}")
print(f"Testing F1-score: {test_f1:.4f}")

fpr, tpr, _ = roc_curve(y_test, y_test_probs)


# Compute AUC
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], "k--")  # Random guess line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.show()

print(f"Area Under the Curve (AUC): {roc_auc:.4f}")

#-------------------------------------------------------------------------------
# c


# Re-declare x, y but for all classes
band_indices = [4, 24, 44, 58, 84]  # Blue, Green, Red, Red-Edge, NIR

# Extract the selected bands
selected_bands = pavia[:, :, band_indices]  # Shape: (610, 340, 5)

X = selected_bands.reshape(-1, 5)
y = pavia_gt.flatten()

# Remove class 0
mask = y != 0
X = X[mask]
y = y[mask]

print(f"Filtered dataset size: {X.shape[0]}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Check class distribution in train and test sets
train_unique, train_counts = np.unique(y_train, return_counts=True)
test_unique, test_counts = np.unique(y_test, return_counts=True)

print(f"Training Class Distribution: {dict(zip(train_unique, train_counts))}")
print(f"Testing Class Distribution: {dict(zip(test_unique, test_counts))}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features standardized check")


# Train XGBoost classifier
y_train_adj = y_train - 1
y_test_adj = y_test - 1

# Now train XGBoost
xgb_model = XGBClassifier(objective="multi:softmax", num_class=9, eval_metric="mlogloss", random_state=42)
xgb_model.fit(X_train_scaled, y_train_adj)

# Predict and shift labels back to original range
y_test_pred_adj = xgb_model.predict(X_test_scaled)
y_test_pred = y_test_pred_adj + 1

print("Classification Report (Test Set):")
print(classification_report(y_test, y_test_pred, digits=4))

cm = confusion_matrix(y_test, y_test_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

feature_importance = xgb_model.feature_importances_

# Plot feature importance
plt.figure(figsize=(8, 6))
plt.bar(["Blue", "Green", "Red", "Red-Edge", "NIR"], feature_importance)
plt.xlabel("Spectral Band")
plt.ylabel("Feature Importance Score")
plt.title("XGBoost Feature Importance")
plt.show()

# BALANCED MODEL
from imblearn.over_sampling import SMOTE

# Apply SMOTE to create a balanced dataset
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# Check new class distribution
train_balanced_unique, train_balanced_counts = np.unique(y_train_balanced, return_counts=True)
print(f"Balanced Training Class Distribution: {dict(zip(train_balanced_unique, train_balanced_counts))}")

y_train_balanced_adj = y_train_balanced - 1
y_test_adj = y_test - 1

# Train XGBoost with balanced data
xgb_balanced = XGBClassifier(objective="multi:softmax", num_class=9, eval_metric="mlogloss", random_state=42)
xgb_balanced.fit(X_train_balanced, y_train_balanced_adj)

# Make predictions
y_balanced_pred_adj = xgb_balanced.predict(X_test_scaled)
y_balanced_pred = y_balanced_pred_adj + 1


# Report new metrics
print("Classification Report (Balanced Model):")
print(classification_report(y_test, y_balanced_pred, digits=4))

