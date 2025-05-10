# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 1: Load dataset
df = pd.read_csv('Pollutant_Radar.csv')

# Step 2: Explore and clean data
print("First 5 rows:\n", df.head())
print("\nMissing values:\n", df.isnull().sum())
df.dropna(inplace=True)  # Drop rows with missing values

# Step 3: Encode categorical variables if any
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 4: Feature/Target split
# Assuming there's a target column like 'pollutant_id' or similar
# Replace 'pollutant_id' with actual target column name from your dataset
target_column = 'pollutant_id' # Changed target_column to 'pollutant_id'
X = df.drop(columns=[target_column])
y = df[target_column]

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Train with Random Forest and XGBoost

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_preds = rf_model.predict(X_test_scaled)

# XGBoost
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train_scaled, y_train)
xgb_preds = xgb_model.predict(X_test_scaled)

# Step 8: Evaluation
print("\nRandom Forest Classification Report:\n", classification_report(y_test, rf_preds))
print("XGBoost Classification Report:\n", classification_report(y_test, xgb_preds))

# Step 9: Confusion Matrix
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test, rf_preds), annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Confusion Matrix")

plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_test, xgb_preds), annot=True, fmt='d', cmap='Greens')
plt.title("XGBoost Confusion Matrix")

plt.tight_layout()
plt.show()

# Step 10: Feature Importance (Updated)
plt.figure(figsize=(12, 8))
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=True)

plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.title("XGBoost Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()