"""
Iris Species Classification using Decision Tree
================================================
This script demonstrates a complete ML pipeline including:
- Data loading and exploration
- Preprocessing (missing values, label encoding)
- Model training (Decision Tree)
- Model evaluation (accuracy, precision, recall)
- Data visualization and results saved as images
"""

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

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
df['species_name'] = df['species'].map({i: name for i, name in enumerate(iris.target_names)})

# Display basic information about the dataset
print(f"\nDataset Shape: {df.shape}")
print(f"Features: {iris.feature_names}")
print(f"Target Classes: {iris.target_names}")
print("\nFirst 5 rows of the dataset:")
print(df.head())

# ============================================================================
# STEP 2: Exploratory Data Analysis with Visualizations
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

# Visualization 1: Class Distribution
print("\nGenerating Visualization 1: Class Distribution...")
fig, ax = plt.subplots(figsize=(10, 6))
species_counts = df['species_name'].value_counts()
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = ax.bar(species_counts.index, species_counts.values, color=colors, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Species', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Iris Species in Dataset', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)
# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('01_class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: 01_class_distribution.png")

# Visualization 2: Feature Distributions
print("Generating Visualization 2: Feature Distributions...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distribution of Features by Species', fontsize=16, fontweight='bold', y=1.00)
colors_species = {'setosa': '#FF6B6B', 'versicolor': '#4ECDC4', 'virginica': '#45B7D1'}

for idx, feature in enumerate(iris.feature_names):
    ax = axes[idx // 2, idx % 2]
    for species in iris.target_names:
        data = df[df['species_name'] == species][feature]
        ax.hist(data, alpha=0.6, label=species, bins=15, color=colors_species[species], edgecolor='black')
    ax.set_xlabel(feature, fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('02_feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: 02_feature_distributions.png")

# Visualization 3: Pairplot
print("Generating Visualization 3: Feature Pairplot...")
pairplot = sns.pairplot(df, hue='species_name', palette=colors_species, 
                        diag_kind='kde', markers=['o', 's', 'D'],
                        plot_kws={'alpha': 0.6, 'edgecolor': 'black', 'linewidth': 0.5})
pairplot.fig.suptitle('Pairwise Relationships Between Features', y=1.01, fontsize=16, fontweight='bold')
plt.savefig('03_feature_pairplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: 03_feature_pairplot.png")

# Visualization 4: Correlation Heatmap
print("Generating Visualization 4: Correlation Heatmap...")
fig, ax = plt.subplots(figsize=(10, 8))
correlation_matrix = df[iris.feature_names].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1, center=0, ax=ax)
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('04_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: 04_correlation_heatmap.png")

# Visualization 5: Box Plots
print("Generating Visualization 5: Feature Box Plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Box Plots of Features by Species', fontsize=16, fontweight='bold', y=1.00)

for idx, feature in enumerate(iris.feature_names):
    ax = axes[idx // 2, idx % 2]
    df.boxplot(column=feature, by='species_name', ax=ax, 
               patch_artist=True, 
               boxprops=dict(facecolor='lightblue', edgecolor='black'),
               medianprops=dict(color='red', linewidth=2),
               whiskerprops=dict(color='black'),
               capprops=dict(color='black'))
    ax.set_xlabel('Species', fontsize=11, fontweight='bold')
    ax.set_ylabel(feature, fontsize=11, fontweight='bold')
    ax.set_title(f'{feature}', fontsize=12, fontweight='bold')
    ax.get_figure().suptitle('')  # Remove automatic title

plt.suptitle('Box Plots of Features by Species', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('05_feature_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: 05_feature_boxplots.png")

# ============================================================================
# STEP 3: Data Preprocessing
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: Data Preprocessing")
print("=" * 70)

# Handle missing values (if any)
if df.isnull().sum().sum() > 0:
    print("\nHandling missing values...")
    for col in df.columns[:-2]:  # Exclude target columns
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  - Filled {col} with median: {median_val:.2f}")
else:
    print("\nNo missing values found in the dataset!")

# Label Encoding
print("\nLabel Encoding:")
print(f"  Original target values: {df['species'].unique()}")
label_mapping = {i: name for i, name in enumerate(iris.target_names)}
print(f"  Label mapping: {label_mapping}")

# ============================================================================
# STEP 4: Split Features and Target
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: Preparing Data for Training")
print("=" * 70)

# Separate features (X) and target (y)
X = df[iris.feature_names]
y = df['species']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
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
dt_classifier = DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,
    random_state=42
)

# Train the model
print("\nTraining the model...")
dt_classifier.fit(X_train, y_train)
print("Training completed!")

print(f"\nDecision Tree Information:")
print(f"  - Max depth: {dt_classifier.get_depth()}")
print(f"  - Number of leaves: {dt_classifier.get_n_leaves()}")
print(f"  - Number of features: {dt_classifier.n_features_in_}")

# Visualization 6: Decision Tree Structure
print("\nGenerating Visualization 6: Decision Tree Structure...")
fig, ax = plt.subplots(figsize=(20, 12))
plot_tree(dt_classifier, 
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True, 
          rounded=True,
          fontsize=10,
          ax=ax)
plt.title('Decision Tree Structure', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('06_decision_tree_structure.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: 06_decision_tree_structure.png")

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
# STEP 7: Model Evaluation with Visualizations
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

# Calculate precision
train_precision = precision_score(y_train, y_train_pred, average='macro')
test_precision = precision_score(y_test, y_test_pred, average='macro')

print(f"\nPrecision Scores (Macro Average):")
print(f"  - Training Precision: {train_precision:.4f}")
print(f"  - Testing Precision: {test_precision:.4f}")

# Calculate recall
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
cm = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix (Test Set):")
print(cm)

# Visualization 7: Confusion Matrix Heatmap
print("\nGenerating Visualization 7: Confusion Matrix...")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names,
            yticklabels=iris.target_names,
            cbar_kws={'label': 'Count'},
            linewidths=2, linecolor='black',
            square=True, ax=ax)
ax.set_xlabel('Predicted Species', fontsize=12, fontweight='bold')
ax.set_ylabel('Actual Species', fontsize=12, fontweight='bold')
ax.set_title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('07_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: 07_confusion_matrix.png")

# Visualization 8: Performance Metrics Comparison
print("Generating Visualization 8: Performance Metrics...")
metrics_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall'] * 2,
    'Score': [train_accuracy, train_precision, train_recall, 
              test_accuracy, test_precision, test_recall],
    'Set': ['Training'] * 3 + ['Testing'] * 3
}
metrics_df = pd.DataFrame(metrics_data)

fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(len(['Accuracy', 'Precision', 'Recall']))
width = 0.35

train_scores = [train_accuracy, train_precision, train_recall]
test_scores = [test_accuracy, test_precision, test_recall]

bars1 = ax.bar(x - width/2, train_scores, width, label='Training', 
               color='#4ECDC4', edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, test_scores, width, label='Testing', 
               color='#FF6B6B', edgecolor='black', linewidth=1.5)

ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Metrics Comparison', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(['Accuracy', 'Precision', 'Recall'])
ax.legend(fontsize=11)
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('08_performance_metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved: 08_performance_metrics.png")

# ============================================================================
# STEP 8: Feature Importance
# ============================================================================
print("\n" + "=" * 70)
print("STEP 8: Feature Importance Analysis")
print("=" * 70)

# Get feature importance
feature_importance = dt_classifier.feature_importances_

# Create DataFrame
importance_df = pd.DataFrame({
    'Feature': iris.feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\nFeature Importance Ranking:")
for idx, row in importance_df.iterrows():
    print(f"  {row['Feature']:30s}: {row['Importance']:.4f}")

# Visualization 9: Feature Importance
print("\nGenerating Visualization 9: Feature Importance...")
fig, ax = plt.s