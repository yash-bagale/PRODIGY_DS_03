# -----------------------------
# Import Libraries
# -----------------------------
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -----------------------------
# Load Dataset
# -----------------------------
# Use bank-full.csv (semicolon separated file)
df = pd.read_csv("bank-full.csv", sep=';')

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())


# -----------------------------
# Data Preprocessing
# -----------------------------

# Convert target variable (yes/no → 1/0)
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# Encode categorical columns
label_encoders = {}

for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le


# -----------------------------
# Feature Selection
# -----------------------------
X = df.drop('y', axis=1)
y = df['y']


# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------------
# Build Decision Tree Model
# -----------------------------
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)


# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test)


# -----------------------------
# Model Evaluation
# -----------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# -----------------------------
# Feature Importance
# -----------------------------
importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

plt.figure()
importances.head(10).plot(kind='bar')
plt.title("Top 10 Important Features")
plt.show()

print("\nModel Training Completed Successfully.")
