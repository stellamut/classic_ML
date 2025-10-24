"""
Iris Species Classification - Interactive Streamlit App
========================================================
An interactive web application for exploring and predicting iris species
using a Decision Tree classifier with real-time visualizations.
"""

import streamlit as st
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Iris Species Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF6B6B;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<p class="main-header">🌸 Iris Species Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Machine Learning powered species prediction and data exploration</p>', unsafe_allow_html=True)

# ============================================================================
# Load and Cache Data
# ============================================================================
@st.cache_data
def load_data():
    """Load and prepare the Iris dataset"""
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species_name'] = df['species'].map({i: name for i, name in enumerate(iris.target_names)})
    return df, iris

# ============================================================================
# Train Model
# ============================================================================
@st.cache_resource
def train_model(max_depth, test_size):
    """Train the Decision Tree classifier"""
    df, iris = load_data()
    
    X = df[iris.feature_names]
    y = df['species']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    model = DecisionTreeClassifier(
        criterion='gini',
        max_depth=max_depth,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    return model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, iris

# Load data
df, iris = load_data()

# ============================================================================
# Sidebar - Configuration
# ============================================================================
st.sidebar.header("⚙️ Model Configuration")

max_depth = st.sidebar.slider(
    "Maximum Tree Depth",
    min_value=1,
    max_value=10,
    value=4,
    help="Controls the depth of the decision tree"
)

test_size = st.sidebar.slider(
    "Test Set Size (%)",
    min_value=10,
    max_value=40,
    value=20,
    help="Percentage of data used for testing"
) / 100

st.sidebar.markdown("---")
st.sidebar.header("📊 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Home & Prediction", "📈 Data Explorer", "🎯 Model Performance", "🌳 Decision Tree"]
)

# Train model with selected parameters
model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, iris = train_model(max_depth, test_size)

# ============================================================================
# PAGE 1: Home & Prediction
# ============================================================================
if page == "🏠 Home & Prediction":
    
    # Key metrics at the top
    col1, col2, col3, col4 = st.columns(4)
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='macro')
    test_recall = recall_score(y_test, y_test_pred, average='macro')
    
    with col1:
        st.metric("Test Accuracy", f"{test_accuracy:.2%}")
    with col2:
        st.metric("Precision", f"{test_precision:.2%}")
    with col3:
        st.metric("Recall", f"{test_recall:.2%}")
    with col4:
        st.metric("Dataset Size", f"{len(df)} samples")
    
    st.markdown("---")
    
    # Prediction section
    st.header("🔮 Make a Prediction")
    st.write("Adjust the sliders to input flower measurements and get a species prediction:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sepal_length = st.slider(
            "Sepal Length (cm)",
            float(df[iris.feature_names[0]].min()),
            float(df[iris.feature_names[0]].max()),
            float(df[iris.feature_names[0]].mean()),
            0.1
        )
        
        sepal_width = st.slider(
            "Sepal Width (cm)",
            float(df[iris.feature_names[1]].min()),
            float(df[iris.feature_names[1]].max()),
            float(df[iris.feature_names[1]].mean()),
            0.1
        )
    
    with col2:
        petal_length = st.slider(
            "Petal Length (cm)",
            float(df[iris.feature_names[2]].min()),
            float(df[iris.feature_names[2]].max()),
            float(df[iris.feature_names[2]].mean()),
            0.1
        )
        
        petal_width = st.slider(
            "Petal Width (cm)",
            float(df[iris.feature_names[3]].min()),
            float(df[iris.feature_names[3]].max()),
            float(df[iris.feature_names[3]].mean()),
            0.1
        )
    
    # Make prediction
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]
    predicted_species = iris.target_names[prediction]
    
    # Display prediction result
    st.markdown("---")
    st.subheader("Prediction Result")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        <div style='background-color: #4ECDC4; padding: 2rem; border-radius: 1rem; text-align: center;'>
            <h2 style='color: white; margin: 0;'>🌸 {predicted_species.upper()}</h2>
            <p style='color: white; margin: 0.5rem 0 0 0; font-size: 1.2rem;'>
                Confidence: {prediction_proba[prediction]:.1%}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Probability chart
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars = ax.barh(iris.target_names, prediction_proba, color=colors, edgecolor='black', linewidth=2)
        ax.set_xlabel('Probability', fontsize=12, fontweight='bold')
        ax.set_title('Prediction Probabilities', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1])
        
        # Add percentage labels
        for i, (bar, prob) in enumerate(zip(bars, prediction_proba)):
            ax.text(prob, bar.get_y() + bar.get_height()/2, 
                   f' {prob:.1%}', 
                   va='center', fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Dataset preview
    st.markdown("---")
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

# ============================================================================
# PAGE 2: Data Explorer
# ============================================================================
elif page == "📈 Data Explorer":
    
    st.header("Data Exploration & Visualization")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "🔗 Correlations", "📦 Box Plots", "📉 Statistics"])
    
    with tab1:
        st.subheader("Feature Distributions by Species")
        
        col1, col2 = st.columns(2)
        
        with col1:
            feature1 = st.selectbox("Select Feature 1", iris.feature_names, index=0)
        with col2:
            feature2 = st.selectbox("Select Feature 2", iris.feature_names, index=2)
        
        # Histograms
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            colors_species = {'setosa': '#FF6B6B', 'versicolor': '#4ECDC4', 'virginica': '#45B7D1'}
            for species in iris.target_names:
                data = df[df['species_name'] == species][feature1]
                ax.hist(data, alpha=0.6, label=species, bins=15, 
                       color=colors_species[species], edgecolor='black')
            ax.set_xlabel(feature1, fontsize=12, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax.set_title(f'Distribution of {feature1}', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            for species in iris.target_names:
                data = df[df['species_name'] == species][feature2]
                ax.hist(data, alpha=0.6, label=species, bins=15, 
                       color=colors_species[species], edgecolor='black')
            ax.set_xlabel(feature2, fontsize=12, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax.set_title(f'Distribution of {feature2}', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        # Scatter plot
        st.subheader("Scatter Plot")
        fig, ax = plt.subplots(figsize=(10, 6))
        for species in iris.target_names:
            species_data = df[df['species_name'] == species]
            ax.scatter(species_data[feature1], species_data[feature2], 
                      label=species, alpha=0.6, s=100, 
                      color=colors_species[species], edgecolor='black', linewidth=1)
        ax.set_xlabel(feature1, fontsize=12, fontweight='bold')
        ax.set_ylabel(feature2, fontsize=12, fontweight='bold')
        ax.set_title(f'{feature1} vs {feature2}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    with tab2:
        st.subheader("Feature Correlation Heatmap")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        correlation_matrix = df[iris.feature_names].corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                   square=True, linewidths=2, cbar_kws={"shrink": 0.8},
                   vmin=-1, vmax=1, center=0, ax=ax)
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
        st.pyplot(fig)
        plt.close()
        
        st.info("💡 Strong positive correlations (red) indicate features that increase together. Petal measurements show high correlation!")
    
    with tab3:
        st.subheader("Box Plots by Species")
        
        selected_feature = st.selectbox("Select Feature for Box Plot", iris.feature_names)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        df.boxplot(column=selected_feature, by='species_name', ax=ax,
                  patch_artist=True,
                  boxprops=dict(facecolor='lightblue', edgecolor='black'),
                  medianprops=dict(color='red', linewidth=2),
                  whiskerprops=dict(color='black'),
                  capprops=dict(color='black'))
        ax.set_xlabel('Species', fontsize=12, fontweight='bold')
        ax.set_ylabel(selected_feature, fontsize=12, fontweight='bold')
        ax.set_title(f'Box Plot: {selected_feature} by Species', fontsize=14, fontweight='bold')
        plt.suptitle('')
        st.pyplot(fig)
        plt.close()
        
        st.info("💡 Box plots show the median (red line), quartiles (box), and outliers for each species.")
    
    with tab4:
        st.subheader("Statistical Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Overall Statistics**")
            st.dataframe(df[iris.feature_names].describe(), use_container_width=True)
        
        with col2:
            st.write("**Class Distribution**")
            class_dist = df['species_name'].value_counts()
            st.dataframe(class_dist, use_container_width=True)
            
            # Pie chart
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            ax.pie(class_dist.values, labels=class_dist.index, autopct='%1.1f%%',
                  colors=colors, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
            ax.set_title('Class Distribution', fontsize=14, fontweight='bold', pad=20)
            st.pyplot(fig)
            plt.close()

# ============================================================================
# PAGE 3: Model Performance
# ============================================================================
elif page == "🎯 Model Performance":
    
    st.header("Model Performance Analysis")
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_precision = precision_score(y_train, y_train_pred, average='macro')
    test_precision = precision_score(y_test, y_test_pred, average='macro')
    train_recall = recall_score(y_train, y_train_pred, average='macro')
    test_recall = recall_score(y_test, y_test_pred, average='macro')
    
    # Metrics comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Performance Metrics")
        
        metrics_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall'],
            'Training': [f"{train_accuracy:.4f}", f"{train_precision:.4f}", f"{train_recall:.4f}"],
            'Testing': [f"{test_accuracy:.4f}", f"{test_precision:.4f}", f"{test_recall:.4f}"]
        }
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        # Performance bar chart
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.arange(3)
        width = 0.35
        
        train_scores = [train_accuracy, train_precision, train_recall]
        test_scores = [test_accuracy, test_precision, test_recall]
        
        bars1 = ax.bar(x - width/2, train_scores, width, label='Training', 
                      color='#4ECDC4', edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, test_scores, width, label='Testing', 
                      color='#FF6B6B', edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Accuracy', 'Precision', 'Recall'])
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("🎯 Confusion Matrix")
        
        cm = confusion_matrix(y_test, y_test_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=iris.target_names,
                   yticklabels=iris.target_names,
                   cbar_kws={'label': 'Count'},
                   linewidths=2, linecolor='black',
                   square=True, ax=ax, annot_kws={"fontsize": 14, "fontweight": "bold"})
        ax.set_xlabel('Predicted Species', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual Species', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
        st.pyplot(fig)
        plt.close()
        
        st.info("💡 Diagonal values show correct predictions. Off-diagonal values indicate misclassifications.")
    
    # Classification report
    st.subheader("📋 Detailed Classification Report")
    
    report = classification_report(y_test, y_test_pred, target_names=iris.target_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
    
    # Per-class performance
    st.subheader("📈 Per-Class Performance")
    
    precision_per_class = precision_score(y_test, y_test_pred, average=None)
    recall_per_class = recall_score(y_test, y_test_pred, average=None)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(iris.target_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, precision_per_class, width, label='Precision',
                  color='#45B7D1', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, recall_per_class, width, label='Recall',
                  color='#FFA07A', edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Species', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Precision and Recall by Species', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(iris.target_names)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    st.pyplot(fig)
    plt.close()

# ============================================================================
# PAGE 4: Decision Tree
# ============================================================================
elif page == "🌳 Decision Tree":
    
    st.header("Decision Tree Visualization")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tree Depth", model.get_depth())
    with col2:
        st.metric("Number of Leaves", model.get_n_leaves())
    with col3:
        st.metric("Number of Features", model.n_features_in_)
    
    st.markdown("---")
    
    # Feature importance
    st.subheader("🎯 Feature Importance")
    
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': iris.feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_imp = ['#FF6B6B' if i == len(importance_df)-1 else '#4ECDC4' for i in range(len(importance_df))]
    bars = ax.barh(importance_df['Feature'], importance_df['Importance'],
                  color=colors_imp, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Features', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance Ranking', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    for bar, val in zip(bars, importance_df['Importance']):
        ax.text(val, bar.get_y() + bar.get_height()/2,
               f' {val:.4f}', va='center', fontweight='bold', fontsize=11)
    
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    
    # Tree structure
    st.subheader("🌳 Tree Structure")
    st.info("⚠️ Note: Large trees may take time to render. Consider reducing max depth for better visualization.")
    
    if st.button("Generate Tree Visualization", type="primary"):
        with st.spinner("Generating tree visualization..."):
            fig, ax = plt.subplots(figsize=(20, 12))
            plot_tree(model,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True,
                     rounded=True,
                     fontsize=10,
                     ax=ax)
            plt.title('Decision Tree Structure', fontsize=16, fontweight='bold', pad=20)
            st.pyplot(fig)
            plt.close()

# ============================================================================
# Footer
# ============================================================================
st.sidebar.markdown("---")
st.sidebar.info("""
**About this App**

This interactive application demonstrates machine learning classification using the famous Iris dataset.

**Features:**
- Real-time predictions
- Interactive visualizations
- Model performance metrics
- Decision tree analysis

**Tech Stack:**
- Streamlit
- Scikit-learn
- Matplotlib/Seaborn

""")

