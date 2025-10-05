# **🪐 Exoplanet Disposition Classifier (LightGBM)**

## **Project Name: Exoplanet Classifier**

### **1\. Executive Summary**

This project delivers a complete, AI-powered pipeline for classifying exoplanet candidates into **CONFIRMED**, **CANDIDATE**, or **FALSE POSITIVE** outcomes. Leveraging observational data from the Kepler and TESS missions, the solution utilizes a highly optimized **LightGBM gradient-boosting model** trained on a carefully engineered set of **27 physical and uncertainty features**. The system automates the astronomical vetting process, significantly increasing the efficiency and statistical robustness of exoplanet discovery.

### **2\. Technical Deep Dive and Methodology**

The core strength of this project lies in its meticulous feature engineering and robust handling of scientific data challenges.

#### **2.1. Feature Engineering (The 27 Features)**

The model's superior performance is built on a custom set of features designed to capture the physics and data quality of each transit event. This includes:

| Feature Category | Example Features | Purpose |
| :---- | :---- | :---- |
| **Data Quality (Uncertainty)** | combined\_uncertainty\_score, transit\_duration\_uncertainty | Measures the reliability of the underlying measurement. The high rank of these features confirms the model prioritizes trustworthy data. |
| **Physical Ratios** | planet\_to\_star\_radius\_ratio, transit\_duration\_to\_period\_ratio | Provides context on the size and geometry of the orbit, helping the model distinguish between a large, confirmed planet and a false grazing eclipse. |
| **Transformations** | log\_OrbitalPeriod\_days, log\_TransitDepth\_ppm | Compresses features with vast magnitude ranges, making linear patterns more detectable by the tree-based model. |

#### **2.2. Preprocessing Pipeline (preprocess\_data.py)**

A robust, self-contained pipeline was developed to prepare any raw uploaded data (KOI or TESS format) for the model. This is critical for real-world deployment:

* **Mapping & Alignment**: Automatically renames raw columns (e.g., pl\_orbper to OrbitalPeriod\_days) to ensure input consistency.  
* **Feature Creation**: Recalculates all 27 engineered features.  
* **Imputation**: Uses a SimpleImputer(strategy='median') to handle missing values, preventing the system from crashing on incomplete data.

#### **2.3. Model Selection and Optimization**

* **Model**: **LightGBM Classifier**. Chosen for its superior efficiency, speed, and proven performance on structured, high-dimensional data over alternatives like Random Forest.  
* **Class Imbalance Mitigation**: The extreme imbalance (few **CONFIRMED** planets vs. many **CANDIDATE** or **FALSE POSITIVE** entries) was addressed using the **SMOTE** oversampling technique to balance the training data, allowing the model to effectively learn the minority class signatures.  
* **Hyperparameter Tuning**: Performance was maximized using a comprehensive **Grid Search** to find the optimal combination of parameters (n\_estimators, max\_depth, learning\_rate, etc.).

### **3\. Final Performance and Validation**

The final model was validated against an independent test set, demonstrating excellent generalization capabilities:

| Metric | Score | Insight |
| :---- | :---- | :---- |
| **Overall Accuracy** | **82.3%** | The final accuracy achieved by the tuned LightGBM model. |
| **Macro Avg F1-Score** | **0.80** | Demonstrates balanced performance; the model is not biased toward the majority classes. |

| Class | Precision (Purity) | Recall (Completeness) | F1-Score |
| :---- | :---- | :---- | :---- |
| **CONFIRMED (Minority)** | 0.65 | **0.69** | 0.67 |
| **FALSE POSITIVE** | 0.87 | 0.86 | 0.86 |
| **CANDIDATE** | 0.87 | 0.88 | 0.88 |

* **Validation Insight**: The high Recall (69%) for the **CONFIRMED** class is the most critical success factor, indicating the model is highly effective at identifying true planets that might otherwise be missed.

### **4\. Tools and Dependencies**

* **Language**: Python 3.10+  
* **Core Libraries**: pandas, numpy,Seaborn, scikit-learn, lightgbm,xgboost,RandomForest,imbalanced-learn, joblib.  
* **Application**: **Streamlit** (for the interactive web interface).  
* **Development**: Jupyter Notebook (Model.ipynb) and custom Python modules.  
* **Data Source**: NASA Exoplanet Archive (Kepler KOI and TESS TOI data).