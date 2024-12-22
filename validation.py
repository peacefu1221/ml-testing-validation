
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]
df = pd.read_csv(url, names=columns, na_values='?')

# Data preprocessing
# Handle missing values
df.fillna(df.mean(), inplace=True)

# Split features and target
X = df.drop('target', axis=1)
y = (df['target'] > 0).astype(int)  # Binary classification: presence of heart disease

# Normalize the features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Initial performance evaluation
y_pred = model.predict(X_test)
initial_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1-Score': f1_score(y_test, y_pred)
}
print("Initial Metrics:", initial_metrics)

# k-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='accuracy')
print("k-Fold Cross-Validation Accuracy:", np.mean(cv_scores))

# Bootstrapping
bootstrap_scores = []
np.random.seed(42)
for _ in range(100):
    indices = np.random.choice(range(len(X_scaled)), size=len(X_scaled), replace=True)
    X_bootstrap, y_bootstrap = X_scaled[indices], y.iloc[indices]
    model.fit(X_bootstrap, y_bootstrap)
    y_pred_bootstrap = model.predict(X_test)
    bootstrap_scores.append(accuracy_score(y_test, y_pred_bootstrap))

print("Bootstrapping Accuracy:", np.mean(bootstrap_scores))

# Save the results
results = {
    'Initial Metrics': initial_metrics,
    'k-Fold CV Accuracy': np.mean(cv_scores),
    'Bootstrapping Accuracy': np.mean(bootstrap_scores)
}
pd.DataFrame([results]).to_csv("results.csv", index=False)

print("Results saved to 'results.csv'")
