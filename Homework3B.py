import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.neural_network import MLPRegressor

# Problem 2
# a
data = np.load('landis_chlorophyl_regression.npy')
data_gt = np.load('landis_chlorophyl_regression_gt.npy')

print(data.shape)
print(data_gt.shape)

band_labels = [
    "Blue (490nm)",
    "Green (580nm)",
    "Yellow (600nm)",
    "Orange (620nm)",
    "Red 1 (650nm)",
    "Red 2 (665nm)",
    "Red Edge 1 (705nm)",
    "Red Edge 2 (740nm)",
    "NIR_Broad (843nm)",
    "NIR1 (865nm)"
]

df = pd.DataFrame(data, columns=band_labels)
df['Chlorophyll'] = data_gt

# Basic Info
print(df.info())
print(df.describe())

# Missing Values
print("Missing Values:\n", df.isnull().sum())

def correlation_matrix(args):

    bands_flattened = data.reshape(-1, data.shape[-1])
    corr_matrix = np.corrcoef(bands_flattened, rowvar=False)

    return corr_matrix

corr_matrix = correlation_matrix(data)

plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, cmap='coolwarm', vmin=0, vmax=1)
plt.colorbar(label='Correlation Coefficient')
plt.title("Correlation Matrix")
plt.xlabel("Bands")
plt.ylabel("Bands")
plt.xticks(range(data.shape[-1]), range(1, data.shape[-1] + 1))
plt.yticks(range(data.shape[-1]), range(1, data.shape[-1] + 1))
plt.show()

# Multicoliniarity
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[band_labels])
vif_data = pd.DataFrame()
vif_data['Feature'] = band_labels
vif_data['VIF'] = [variance_inflation_factor(X_scaled, i) for i in range(len(band_labels))]
print("VIF Data:\n", vif_data)

# Pairplot
sns.pairplot(df, vars=band_labels[:11], diag_kind='kde')
plt.show()

#------------------------------------------------------------------------
# b
X = df[band_labels]
y = df['Chlorophyll']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluation Metrics
def evaluate_model(y_true, y_pred, dataset_type="Training"):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    residuals = y_true - y_pred
    std_residuals = np.std(residuals)
    return mae, r2, std_residuals

mae_train, r2_train, std_train = evaluate_model(y_train, y_train_pred, "Training")
mae_test, r2_test, std_test = evaluate_model(y_test, y_test_pred, "Testing")

# Regression Plots
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_train, y=y_train_pred, alpha=0.5)
plt.xlabel("Actual Chlorophyll")
plt.ylabel("Predicted Chlorophyll")
plt.title("Training Set: Actual vs Predicted")
plt.text(min(y_train), max(y_train_pred) -15, f"MAE: {mae_train:.4f}\nR²: {r2_train:.4f}\nStd Residuals: {std_train:.4f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.5)
plt.xlabel("Actual Chlorophyll")
plt.ylabel("Predicted Chlorophyll")
plt.title("Testing Set: Actual vs Predicted")
plt.text(min(y_test), max(y_test_pred) - 15, f"MAE: {mae_test:.4f}\nR²: {r2_test:.4f}\nStd Residuals: {std_test:.4f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
plt.show()

# Residual Plots
plt.figure(figsize=(10, 6))
sns.histplot(y_train - y_train_pred, bins=30, kde=True)
plt.title("Training Residuals Distribution")
plt.text(min(y_train - y_train_pred), max(plt.ylim()) - 25, f"Std Residuals: {std_train:.4f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(y_test - y_test_pred, bins=30, kde=True)
plt.title("Testing Residuals Distribution")
plt.text(min(y_test - y_test_pred), max(plt.ylim()) - 5, f"Std Residuals: {std_test:.4f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
plt.show()

#-------------------------------------------------------------------
# c

train_scores = []
components_range = range(1, 11)

for n_components in components_range:
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_train, y_train)
    train_r2 = pls.score(X_train, y_train)
    train_scores.append(train_r2)

# Identify best-performing component count
best_n_components = components_range[6]
print(f"Best number of components: {best_n_components}")

plt.figure(figsize=(10, 6))
plt.plot(components_range, train_scores, marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Training R² Score")
plt.title("PLSR Training Accuracy vs. Components")
plt.show()

# Train PLSR with the best component count
pls_best = PLSRegression(n_components=best_n_components)
pls_best.fit(X_train, y_train)

# Predictions
y_train_pred = pls_best.predict(X_train).ravel()
y_test_pred = pls_best.predict(X_test).ravel()

# Evaluation Metrics
def evaluate_model(y_true, y_pred, dataset_type="Training"):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    residuals = y_true - y_pred
    std_residuals = np.std(residuals)
    return mae, r2, std_residuals

mae_train, r2_train, std_train = evaluate_model(y_train, y_train_pred, "Training")
mae_test, r2_test, std_test = evaluate_model(y_test, y_test_pred, "Testing")

# Regression Plots
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_train, y=y_train_pred, alpha=0.5)
plt.xlabel("Actual Chlorophyll")
plt.ylabel("Predicted Chlorophyll")
plt.title("PLSR Training Set: Actual vs Predicted")
plt.text(min(y_train), max(y_train_pred) - 15, f"MAE: {mae_train:.4f}\nR²: {r2_train:.4f}\nStd Residuals: {std_train:.4f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.5)
plt.xlabel("Actual Chlorophyll")
plt.ylabel("Predicted Chlorophyll")
plt.title("PLSR Testing Set: Actual vs Predicted")
plt.text(min(y_test), max(y_test_pred) - 15, f"MAE: {mae_test:.4f}\nR²: {r2_test:.4f}\nStd Residuals: {std_test:.4f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
plt.show()

# Residual Plots
plt.figure(figsize=(10, 6))
sns.histplot(y_train - y_train_pred, bins=30, kde=True)
plt.title("PLSR Training Residuals Distribution")
plt.text(min(y_train - y_train_pred), max(plt.ylim()) - 25, f"Std Residuals: {std_train:.4f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(y_test - y_test_pred, bins=30, kde=True)
plt.title("PLSR Testing Residuals Distribution")
plt.text(min(y_test - y_test_pred), max(plt.ylim()) - 5, f"Std Residuals: {std_test:.4f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
plt.show()

#-------------------------------------------------------------------
# d

layers = [1, 2, 3, 4, 5]
results = {}

for num_layers in layers:
    mlp = MLPRegressor(hidden_layer_sizes=(100,) * num_layers, activation='relu', max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)
    y_train_pred = mlp.predict(X_train)
    y_test_pred = mlp.predict(X_test)
    mae_train, r2_train, std_train = evaluate_model(y_train, y_train_pred)
    mae_test, r2_test, std_test = evaluate_model(y_test, y_test_pred)
    results[num_layers] = (mae_train, r2_train, std_train, mae_test, r2_test, std_test)

    # Regression Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.5)
    plt.xlabel("Actual Chlorophyll")
    plt.ylabel("Predicted Chlorophyll")
    plt.title(f"MLP ({num_layers} Layers) - Testing Set: Actual vs Predicted")
    plt.text(min(y_test), max(y_test_pred) - 15, f"MAE: {mae_test:.4f}\nR²: {r2_test:.4f}\nStd Residuals: {std_test:.4f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.show()

    # Residual Plot
    residuals = y_test - y_test_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test_pred, y=residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Chlorophyll")
    plt.ylabel("Residuals")
    plt.title(f"MLP ({num_layers} Layers) - Residual Plot")
    plt.text(min(y_test_pred), max(residuals) - 0.5, f"Std Residuals: {std_test:.4f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.show()
