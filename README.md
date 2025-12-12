# Earthquake Data Analysis and Prediction System
## Project Overview
This is a comprehensive earthquake data analysis and prediction system that integrates data processing, visualization, comparison of multiple machine learning models, and spatial clustering analysis. The system can process earthquake data, extract various features, and use multiple machine learning algorithms for classification, regression, and clustering analysis.

## Motivation
### Why did we choose earthquake data analysis as our topic?

Our choice stems primarily from a natural scientific curiosity and its scientific significance: earthquakes, as an unsolved mystery, are among the deadliest natural disasters. In the 21st century alone, earthquakes have caused over 800,000 deaths. Earthquake prediction is considered the "holy grail" of Earth science. While we know we cannot "solve" this problem, systematically analyzing data and searching for patterns is itself a process of scientific exploration.

### What real problem or question does your project attempt to solve?
We can't help but ask: Are these earthquakes isolated or interconnected? Why do large earthquakes seem to "cluster" together? Are there global earthquake triggering mechanisms that we don't yet understand? Do these regions correspond to the approximate directions of known seismic zones? Can we use algorithms to automatically identify them?

Furthermore, the spatiotemporal "clustering patterns" of this narrative-rich data also intrigue us: Do earthquakes, like weather, have "active seasons" and "quiet seasons"? Are there predictable temporal patterns in seismic activity? For example, are earthquakes more likely to occur at certain times of day or in certain months of the year? The data itself tells a complex story about Earth's dynamics.

## Team Background 
1. Zeng xinwei

2. Li Aiwen

3. Cai Ziyi

4. Zhang Ruqian

## Data Collection
We personally crawled the raw, unprocessed earthquake details data from the China Earthquake Networks Center.
https://www.cenc.ac.cn/earthquake-manage-publish-web/designated-catalogue

### Data Review

## Core Functions
1. Data Processing Module (DataProcessor):  
- Data cleaning and preprocessing  
- Geographic coordinate resolution (latitude and longitude conversion)   
- Magnitude and depth data standardization  
- Advanced feature engineering (time series, spatial, statistical features)

2. Visualization Module (Visualizer):  
- Exploratory Data Analysis (EDA)  
- Multi-dimensional data distribution visualization  
- Model performance comparison charts  
- Spatial distribution heatmaps

3. Model Evaluation Module (ModelEvaluator):  
-  Classification models: Logistic Regression, Random Forest, Decision Tree, SVM, KNN, XGBoost, LightGBM, Gradient Boosting, AdaBoost, Naive Bayes, Neural Networks  
- Regression models: Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, Decision Tree, SVR, KNN, XGBoost, LightGBM, Gradient Boosting, Neural Networks  
- Time series cross-validation  
- Hyperparameter tuning  
- Multiple evaluation metrics   

4. Spatial Analysis Module (SpatialAnalyzer):  
- DBSCAN spatial clustering  
- KMeans clustering  
- Clustering quality assessment  
- Spatial pattern recognition    
##  **How to Run**
### 1. Environment Requirements
- Python 3.8+  
- Works on Windows / macOS / Linux  
- CPU-only compatible

### 2. Install required packages
```
python

pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm openpyxl
```
### 3. Place the data file
Please name your earthquake data as  
```
python
df = pd.read_excel('your file path/earthquake data.xlsx', engine='openpyxl')
```


## **Core Operations / Usage Examples**
## **1. Data Cleaning** 
Converts messy, real-world Chinese data into standardized, analyzable format
```
markdown
Raw Chinese Data ——> Standardized English Data
"2025-12-08 11:23:45" ——> datetime(2025, 12, 8, 11, 23, 45)
"142.35° 41.00°" ——> Longitude=142.35, Latitude=41.00
"30千米" ——> 30.0
"6.6级" ——> 6.6
```

## **2. Feature Engineering**

Earthquake data can be noisy and heterogeneous. To improve model generalization, a set of engineered features is generated to enhance the signal quality and capture underlying geophysical patterns.

### Key Feature Categories

### 1. Statistical Features
- `log_magnitude` — Log transformation stabilizes variance and reduces skewness  
- `depth_squared` — Captures non-linear effects of seismic depth  
- Normalized numerical features (optional) to reduce scale bias  

### 2. Geospatial Features
- `region` — Longitude-based segmentation (West / Middle / East)  
- Cluster-based regional encoding (optional upgrade)  
- Potential future extension: distance to major tectonic plates  

### 3. Temporal & Structural Features
- Hour, day, month, season  
- Moving averages or rolling window statistics  
- Aftershock sequence indicators  

### Why Feature Engineering Matters
- Improves model expressiveness  
- Reduces noise sensitivity  
- Allows models to detect spatial/temporal seismic patterns  
- Improves interpretability (especially with Random Forest importance scores)  

---

## **3. Multi-Model Comparison** 

To assess the predictive performance across different algorithmic families, the project compares several regression models.

### **Models Included:**
### Classification Models (12)
| No. | Model Name | Description |
|-----|------------|-------------|
| 1 | Logistic Regression | Linear classifier for binary/multiclass tasks |
| 2 | Random Forest | Ensemble of decision trees |
| 3 | Decision Tree | Tree-based non-linear classifier |
| 4 | SVM | Support Vector Machine classifier |
| 5 | KNN | K-Nearest Neighbors classifier |
| 6 | XGBoost | Gradient-boosted tree ensemble |
| 7 | LightGBM | Fast gradient boosting tree model |
| 8 | Gradient Boosting | Classical gradient boosting |
| 9 | AdaBoost | Boosting with adaptive weights |
|10 | Naive Bayes | Probabilistic classifier |
|11 | MLP | Neural network classifier |
|12 | Gaussian NB | Gaussian Naive Bayes classifier |
---
### Regression Models (12)

| No. | Model Name | Description |
|-----|------------|-------------|
| 1 | Linear Regression | Ordinary least squares regression |
| 2 | Ridge Regression | L2-regularized regression |
| 3 | Lasso Regression | L1-regularized regression |
| 4 | ElasticNet | Combined L1+L2 regularization |
| 5 | Random Forest Regressor | Tree-based ensemble regressor |
| 6 | Decision Tree Regressor | Non-linear tree regressor |
| 7 | SVR | Support Vector Regression |
| 8 | KNN Regressor | K-Nearest Neighbors regression |
| 9 | XGBoost Regressor | Gradient boosting tree regressor |
|10 | LightGBM Regressor | High-performance boosting regressor |
|11 | Gradient Boosting Regressor | Classical boosting regressor |
|12 | MLP Regressor | Neural network regressor |
---

### Clustering Models (2)

| No. | Model Name | Description |
|-----|------------|-------------|
| 1 | KMeans | Partition-based clustering (k clusters) |
| 2 | DBSCAN | Density-based clustering algorithm |
---

### **Comparison Metrics**
### Classification Metrics (8)  
**Primary Metrics: F1 Score, AUC-ROC**

| No. | Metric | Function | Description |
|-----|--------|-----------|-------------|
| 1 | Accuracy | `accuracy_score` | Ratio of correctly predicted samples |
| 2 | Precision | `precision_score` | TP / (TP + FP), measures prediction purity |
| 3 | Recall | `recall_score` | TP / (TP + FN), measures ability to find positives |
| 4 | F1 Score | `f1_score` | Harmonic mean of precision & recall |
| 5 | AUC-ROC | `roc_auc_score` | Area under ROC curve, measures model separability |
| 6 | Confusion Matrix | `confusion_matrix` | TP/FP/TN/FN summary |
| 7 | Classification Report | `classification_report` | Precision, recall, F1 for each class |
| 8 | Weighted F1 / Macro F1 | `f1_score(..., average=...)` | Supports imbalanced datasets |

---

### Regression Metrics (5)
**Primary Metrics: R² Score, MAE**

| No. | Metric | Function | Description |
|-----|--------|-----------|-------------|
| 1 | MAE | `mean_absolute_error` | Average absolute error; interpretable |
| 2 | MSE | `mean_squared_error` | Penalizes large errors heavily |
| 3 | RMSE | `mean_squared_error` (sqrt) | Root MSE, same unit as target |
| 4 | R² Score | `r2_score` | Variance explained by the model |
| 5 | MAPE (optional) | custom | Mean Absolute Percentage Error |

---

### Clustering Metrics (3)
**Primary Metrics: Silhouette Score, Davies–Bouldin Index (DBI)**

| No. | Metric | Function | Description |
|-----|--------|-----------|-------------|
| 1 | Silhouette Score | `silhouette_score` | Measures cluster separation (higher is better) |
| 2 | Davies–Bouldin Index | `davies_bouldin_score` | Measures cluster similarity (lower is better) |
| 3 | Calinski–Harabasz Index | custom | Measures cluster dispersion (higher is better) |
 
## **4.Hyperparameter Tuning**
This project applies a unified and systematic hyperparameter tuning strategy across all classification, regression, and clustering models. The goal is to ensure fairness, stability, and optimal performance for each model family.

### 1. Search Algorithms
Two search strategies are used depending on model complexity:

- **GridSearchCV** — exhaustive search for small/medium spaces  
- **RandomizedSearchCV** — efficient search for large or heavy models  

All tuning procedures use **5-fold cross-validation** to avoid overfitting and ensure stable model comparison.

---

### 2. Evaluation Metrics for Tuning
Different tasks use different primary metrics:

| Task Type | Primary Metric | Secondary Metrics |
|-----------|----------------|--------------------|
| **Classification** | **F1 Score**, AUC-ROC | Accuracy, Precision, Recall |
| **Regression** | **R² Score**, MAE | MSE, RMSE |
| **Clustering** | **Silhouette Score**, DBI | CH Index |

Corresponding scoring passed into the tuner:
```
python
scoring_classification = "f1_weighted"
scoring_regression = "r2"
```
### 3. Search Space Design
Each model uses a carefully selected hyperparameter set:
- Core structure parameters (e.g., depth, number of estimators)

- Regularization parameters (e.g., C, alpha, l1_ratio)

- Learning parameters (e.g., learning_rate for boosting models)

The design avoids extremely large grids to prevent overfitting and excessive computation.
### 4. Model-Specific Tuning Examples

Only representative models are shown here.
Other models follow the same tuning framework with task-appropriate search spaces.  

**Example 1: Random Forest (Classification)**
```
python
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10]
}
grid = GridSearchCV(RandomForestClassifier(), param_grid,
                    scoring="f1_weighted", cv=5, n_jobs=-1)
grid.fit(X_train, y_train)
best_rf = grid.best_estimator_
```
**Example 2: Random Forest Regressor**
```
python
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None]
}
grid = GridSearchCV(RandomForestRegressor(), param_grid,
                    scoring="r2", cv=5)
grid.fit(X_train, y_train)
best_rf_reg = grid.best_estimator_
```
**Example 3: KMeans (Clustering)**
```
python
best_score = -1
best_k = None
for k in [2, 3, 4, 5, 6]:
    model = KMeans(n_clusters=k)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    if score > best_score:
        best_score = score
        best_k = k
```

 

