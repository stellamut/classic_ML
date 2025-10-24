"""
Iris Species Classification using Decision Tree
================================================
This script demonstrates a complete ML pipeline including:
- Data loading and exploration
- Preprocessing (missing values, label encoding)
- Model training (Decision Tree)
- Model evaluation (accuracy, precision, recall)
"""

# Import required libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: Load the Dataset
# ============================================================================
print("=" * 70)
print("STEP 1: Loading Iris Dataset")
print("=" * 70)

# Load the iris dataset from sklearn
iris = load_iris()

# Create a DataFrame for better data manipulation
df = pd.DataFrame(
    data=iris.data,
    columns=iris.feature_names
)

# Add the target column (species)
df['species'] = iris.target

# Display basic information about the dataset
print(f"\nDataset Shape: {df.shape}")
print(f"Features: {iris.feature_names}")
print(f"Target Classes: {iris.target_names}")
print("\nFirst 5 rows of the dataset:")
print(df.head())

# ============================================================================
# STEP 2: Exploratory Data Analysis
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: Exploratory Data Analysis")
print("=" * 70)

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Display statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Check class distribution
print("\nClass Distribution:")
print(df['species'].value_counts())

# ============================================================================
# STEP 3: Data Preprocessing
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: Data Preprocessing")
print("=" * 70)

# Handle missing values (if any)
# Strategy: Fill missing values with the median of each column
if df.isnull().sum().sum() > 0:
    print("\nHandling missing values...")
    for col in df.columns[:-1]:  # Exclude target column
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  - Filled {col} with median: {median_val:.2f}")
else:
    print("\nNo missing values found in the dataset!")

# Label Encoding for target variable
# Note: In this case, targets are already encoded (0, 1, 2)
# But we'll demonstrate the process for completeness
print("\nLabel Encoding:")
print(f"  Original target values: {df['species'].unique()}")

# The iris dataset already has numeric labels, but let's show the mapping
label_mapping = {i: name for i, name in enumerate(iris.target_names)}
print(f"  Label mapping: {label_mapping}")

# ============================================================================
# STEP 4: Split Features and Target
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: Preparing Data for Training")
print("=" * 70)

# Separate features (X) and target (y)
X = df.drop('species', axis=1)
y = df['species']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # Maintain class distribution in splits
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# ============================================================================
# STEP 5: Train Decision Tree Classifier
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: Training Decision Tree Classifier")
print("=" * 70)

# Initialize the Decision Tree Classifier
# Parameters:
#   - criterion: 'gini' for Gini impurity (default)
#   - max_depth: Maximum depth of the tree (None = unlimited)
#   - random_state: For reproducibility
dt_classifier = DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,
    random_state=42
)

# Train the model
print("\nTraining the model...")
dt_classifier.fit(X_train, y_train)
print("Training completed!")

# Display tree information
print(f"\nDecision Tree Information:")
print(f"  - Max depth: {dt_classifier.get_depth()}")
print(f"  - Number of leaves: {dt_classifier.get_n_leaves()}")
print(f"  - Number of features: {dt_classifier.n_features_in_}")

# ============================================================================
# STEP 6: Make Predictions
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: Making Predictions")
print("=" * 70)

# Predict on training set
y_train_pred = dt_classifier.predict(X_train)

# Predict on testing set
y_test_pred = dt_classifier.predict(X_test)

print("\nPredictions completed!")

# ============================================================================
# STEP 7: Model Evaluation
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: Model Evaluation")
print("=" * 70)

# Calculate accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"\nAccuracy Scores:")
print(f"  - Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"  - Testing Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Calculate precision (macro average for multi-class)
train_precision = precision_score(y_train, y_train_pred, average='macro')
test_precision = precision_score(y_test, y_test_pred, average='macro')

print(f"\nPrecision Scores (Macro Average):")
print(f"  - Training Precision: {train_precision:.4f}")
print(f"  - Testing Precision: {test_precision:.4f}")

# Calculate recall (macro average for multi-class)
train_recall = recall_score(y_train, y_train_pred, average='macro')
test_recall = recall_score(y_test, y_test_pred, average='macro')

print(f"\nRecall Scores (Macro Average):")
print(f"  - Training Recall: {train_recall:.4f}")
print(f"  - Testing Recall: {test_recall:.4f}")

# Detailed classification report
print("\n" + "-" * 70)
print("Detailed Classification Report (Test Set):")
print("-" * 70)
print(classification_report(
    y_test, 
    y_test_pred, 
    target_names=iris.target_names,
    digits=4
))

# Confusion Matrix
print("\nConfusion Matrix (Test Set):")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

# Format confusion matrix with labels
cm_df = pd.DataFrame(
    cm,
    index=[f"True {name}" for name in iris.target_names],
    columns=[f"Pred {name}" for name in iris.target_names]
)
print("\nFormatted Confusion Matrix:")
print(cm_df)

# ============================================================================
# STEP 8: Feature Importance
# ============================================================================
print("\n" + "=" * 70)
print("STEP 8: Feature Importance Analysis")
print("=" * 70)

# Get feature importance from the trained model
feature_importance = dt_classifier.feature_importances_

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({
    'Feature': iris.feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\nFeature Importance Ranking:")
for idx, row in importance_df.iterrows():
    print(f"  {row['Feature']:30s}: {row['Importance']:.4f}")

# ============================================================================
# STEP 9: Example Prediction
# ============================================================================
print("\n" + "=" * 70)
print("STEP 9: Example Prediction")
print("=" * 70)

# Make a prediction on a sample
sample_idx = 0
sample = X_test.iloc[sample_idx].values.reshape(1, -1)
prediction = dt_classifier.predict(sample)
predicted_species = iris.target_names[prediction[0]]
actual_species = iris.target_names[y_test.iloc[sample_idx]]

print(f"\nSample features: {sample[0]}")
print(f"Predicted species: {predicted_species}")
print(f"Actual species: {actual_species}")
print(f"Prediction correct: {predicted_species == actual_species}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
Model: Decision Tree Classifier
Dataset: Iris Species (150 samples, 4 features, 3 classes)

Performance Metrics (Test Set):
  - Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)
  - Precision: {test_precision:.4f}
  - Recall:    {test_recall:.4f}

Most Important Feature: {importance_df.iloc[0]['Feature']}

The model successfully classifies iris species based on flower measurements!
""")