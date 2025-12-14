# **Earthquake Data Analysis and Prediction System**
## **Project Overview**
This is a comprehensive earthquake data analysis and prediction system that integrates data processing, visualization, comparison of multiple machine learning models, and spatial clustering analysis. The system can process earthquake data, extract various features, and use multiple machine learning algorithms for classification, regression, and clustering analysis.

## **Motivation**
### Why did we choose earthquake data analysis as our topic?

Our choice stems primarily from a natural scientific curiosity and its scientific significance: earthquakes, as an unsolved mystery, are among the deadliest natural disasters. In the 21st century alone, earthquakes have caused over 800,000 deaths. Earthquake prediction is considered the "holy grail" of Earth science. While we know we cannot "solve" this problem, systematically analyzing data and searching for patterns is itself a process of scientific exploration.

### What real problem or question does our project attempt to solve?
We can't help but ask: Are these earthquakes isolated or interconnected? Why do large earthquakes seem to "cluster" together? Are there global earthquake triggering mechanisms that we don't yet understand? Do these regions correspond to the approximate directions of known seismic zones? Can we use algorithms to automatically identify them?

Furthermore, the spatiotemporal "clustering patterns" of this narrative-rich data also intrigue us: Do earthquakes, like weather, have "active seasons" and "quiet seasons"? Are there predictable temporal patterns in seismic activity? For example, are earthquakes more likely to occur at certain times of day or in certain months of the year? The data itself tells a complex story about Earth's dynamics.

## **Team Background** 
1. **Zeng xinwei:** Bachelor’s in Data Science and Big Data Technology, with solid foundations in Data Structures, Distributed Computing, and basic Machine Learning. Experienced in end-to-end data processing, proficient in data collection, cleaning, preprocessing, and basic visualization. In this project, she led data collection by independently developing multi-platform web scraping scripts. Conducted raw data verification/cleaning, collaborated in data warehousing and standardization, providing high-quality data for feature engineering and model training.
Her core competencies include data acquisition and scraping, where she is proficient in Python libraries such as Requests, Scrapy, and Selenium, having collected millions of data entries while solving anti-scraping measures and handling data heterogeneity across structured, unstructured, and dynamically loaded sources. In data preprocessing and warehousing, she is skilled in Pandas, NumPy, and database technologies including MySQL and MongoDB, effectively handling data deduplication, missing value imputation, and ensuring efficient storage with standardized data flow. She also possesses basic visualization capabilities using Matplotlib and ECharts, creating informative charts to present data quality assessments and distribution analyses that support team decision-making processes.

2. **Li Aiwen:** Holding a Bachelor's degree in Applied Statistics, possesses a solid theoretical foundation in mathematical statistics and machine learning. With extensive experience in data analysis projects, she is proficient in the entire data science workflow from data cleaning and preprocessing to model building and visualization. In this earthquake analysis project, her responsibilities include the cleaning, conversion, and quality verification of seismic data, constructing temporal and spatial features and completing feature engineering, designing and implementing multiple classification and regression models while conducting performance comparisons and evaluations, as well as developing visual charts and producing data and model visualization outputs.
Her core capabilities directly relevant to this project encompass three key areas. In data processing and cleaning, she is proficient in SQL, Python, and Excel, having semi-automatically cleaned and integrated over 400,000 data points, providing crucial support for seismic data standardization and feature extraction. Regarding predictive model construction, she has participated in empirical research on stock strategies and policy evaluations, demonstrating familiarity with time series and structured data modeling, along with the capability to perform model selection and optimization for classification and regression tasks related to earthquake events using algorithms such as XGBoost, LightGBM, and Random Forest. In visualization and insight generation, she is skilled in Matplotlib, Seaborn, Tableau, and Power BI, capable of creating multidimensional charts and dynamic dashboards that clearly present analytical results to support decision-making processes.

3. **Cai Ziyi:** Holding a Bachelor's degree in Applied Statistics, possesses comprehensive technical expertise spanning the full data science pipeline—from data collection and processing to advanced analysis, predictive modeling, and technical documentation. She is proficient in utilizing programming tools such as Python and R for statistical modeling, machine learning implementation, and data visualization.
Her research background in geographic and environmental systems provides direct relevance to earthquake analysis. This includes previous involvement in studies concerning the multi-party collaborative optimization of land management using evolutionary game theory, urban traffic flow prediction integrating random forests and exponential smoothing methods, and consumer behavior analysis through cluster analysis. These experiences demonstrate her ability to model complex multi-agent interactions within geographic contexts, apply advanced mathematical and computational modeling to spatial problems, and identify behavioral equilibria in intricate systems. In this earthquake analysis project, she is also responsible for developing the technical documentation, including the README file, and providing clear interpretation of the codebase.

4. **Zhang Ruqian:** **1. Machine Learning Experience:** Conducted text sentiment analysis using machine learning models, including data preprocessing, feature extraction, and model training, to classify sentiment polarity in user-generated content. Developed stock trend prediction models by applying time-series analysis and regression algorithms, focusing on feature engineering and performance evaluation. **2. Data Visualization Skills:** Created weather data visualizations to present complex meteorological information through interactive charts and dashboards, enhancing data interpretability for diverse audiences. **3. Collaboration and Communication:** Demonstrates strong communication skills, with experience in clearly presenting technical findings and collaborating within cross-functional teams to align on project goals and deliverables.

### How the team’s background influenced the topic choice?
- Our team’s collective expertise in data science, statistics, and computational modeling naturally converged on earthquake data analysis as our project focus, driven by a unique alignment between our skills and the challenges inherent to this domain. The interdisciplinary nature of our backgrounds—spanning applied statistics, data engineering, geographic systems modeling, and machine learning communication—created a synergistic foundation perfectly suited to tackle the complex, multidimensional problem of seismic pattern recognition and prediction.  

- We were particularly drawn to earthquake analysis because it represents a quintessential data science challenge that demands our combined strengths: the need for robust data acquisition and preprocessing aligns with our data engineering expertise; the requirement for sophisticated temporal and spatial feature engineering matches our statistical and geographic modeling experience; and the necessity to evaluate diverse machine learning models corresponds with our proficiency in comparative algorithm analysis.

- Moreover, our prior involvement in geographically-informed research—such as land management optimization and urban systems prediction—provided direct conceptual relevance, allowing us to approach seismic patterns not merely as abstract data points but as manifestations of complex environmental processes. This domain awareness, coupled with our technical capabilities, enabled us to select a topic that is both methodologically challenging and societally impactful, ensuring that our work contributes meaningfully to the important field of seismic risk assessment while fully utilizing the diverse skill set our team possesses

### Roles and Responsibilities of the Team
- Topic choosing: All Members
- Data Scraping: XinweiZENG
- Data Processing: XinweiZENG, AiwenLI
- Model Construction: AiwenLI
- Data Analysis: AiwenLI, XinweiZENG
- Data Visualization: AiwenLI
- Readme File Making: ZiyiCAI
- PPT Making: RuqianZHANG, ZiyiCAI
- Presentation Making: RuqianZHANG



## **Data Collection**
We crawled data based on **Python** and **Selenium**, aiming to automatically collect detailed seismic data from the directory page designated by the **China Earthquake Networks Center Center**, and export the results to an Excel file. The code is written in a purely function based style, making it easy to read, modify, and extend.
https://www.cenc.ac.cn/earthquake-manage-publish-web/designated-catalogue  
**Collection time**: June 3, 2023- December 9, 2025
### Features
- Automatically opens the CENC designated catalogue page  
- Detects and iterates through earthquake list items  
- Clicks into each earthquake detail page  
- Extracts the following data:
  - List item text (earthquake title)
  - Occurrence time
  - Latitude and longitude
  - Focal depth
  - Magnitude
- Supports limiting the number of records  
- Exports data to Excel (`.xlsx`)
  
### Data Fields
The exported Excel file contains the following columns:

| Field | Description |
|------|-------------|
| 序号 | Record index |
| 列表项文本 | Earthquake title from list page |
| 时间 | Occurrence time |
| 经纬度 | Latitude and longitude |
| 震源深度 | Focal depth |
| 级数 | Magnitude |

### Requirements
- Python 3.8+(we use 3.11.11)
- Google Chrome
- ChromeDriver (must match Chrome version)

### Installation
```bash
pip install selenium pandas openpyxl
```

### Usage
1. Make sure ChromeDriver is installed and added to PATH  
2. Run the script:
```bash
python main.py
```
3. Enter the number of earthquake records to crawl (default: 5)  
4. Choose whether to export the data to Excel  
5. The Excel file will be saved to the Desktop

### Notes
- This project uses Selenium and opens a real browser window  
- Changes in page structure may break XPath or selectors  
- Please use responsibly and avoid excessive requests  


### Disclaimer
This project is intended **for educational and research purposes only**.  
Do not use it for commercial purposes or in violation of the target website’s terms of service.

---
## **Core Functions**
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
## **1. Feature Engineering**

Earthquake data can be noisy and heterogeneous. To improve model generalization, a set of engineered features is generated to enhance the signal quality and capture underlying geophysical patterns.

### 1. Time Data Transformation

- `pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S')` – Convert time string to datetime object.

- Extracted time features:

  - `Year`, `Month`, `Day`, `Hour` ： year, month, day, hour  
  - `Weekday` – day of week (0–6)  
  - `Quarter` – quarter of the year  
  - `Day_of_Year` – day number within the year  
  - `Season` – seasonal feature (Winter, Spring, Summer, Autumn)

### 2. Coordinate Data Transformation

- `str.replace('°', '')` – remove degree symbol  
- Convert longitude/latitude strings  
- `pd.to_numeric()` – convert to numerical type  
- Grid features:  
  - `Longitude_Grid`, `Latitude_Grid` – bin coordinates into 1° grid cells

### 3. Numerical Data Cleaning

- Depth: `str.replace('千米', '').astype(float)` – remove units & convert  
- Magnitude: `str.replace('级', '').astype(float)` – remove units & convert  

- Validity checks:
  - `Longitude_Valid`: range [-180, 180]  
  - `Latitude_Valid`: range [-90, 90]  
  - `Depth_Valid`: > 0  
  - `Magnitude_Valid`: [0, 10]

### 4. Generated Features (Feature Engineering)

### a. Lag Features
- `Magnitude_Lag_{1,2,3}` – previous 1/2/3 earthquake magnitudes  
- `Depth_Lag_{1,2,3}` – previous 1/2/3 depths  
- `Longitude_Lag_{1,2,3}`, `Latitude_Lag_{1,2,3}` – previous 1/2/3 coordinates  
- `Interval_Lag_{1,2,3}` – time interval lag features (1, 2, 3 steps)

### b. Rolling Statistics Features
- `Rolling_Mean_Mag_{7,14,30,90}` – rolling mean magnitude (7/14/30/90 days)  
- `Rolling_Std_Mag_{7,14,30,90}` – rolling std deviation  
- `Rolling_Max_Mag_{7,14,30,90}` – rolling maximum  
- `Rolling_Count_{7,14,30,90}` – rolling earthquake count

### c. Spatial Features
- `Spatial_Cluster` – KMeans spatial clustering (8 clusters)  
- `Cumulative_Region_Count` – cumulative count per seismic region

### d. Combined Features
- `Mag_Depth_Ratio_Lag1` – magnitude/depth ratio of previous event  
- `Energy_Lag1` – previous earthquake energy  
  - Formula: `10^(1.5 * Magnitude + 4.8)`  
- `Time_Interval` – time interval in hours  
- `Is_Aftershock` – binary aftershock label (based on magnitude, distance, time)

### 5. Target Variable Transformation

- `Magnitude_Class` – magnitude categories:  
  - Micro (<3), Minor (3–4), Light (4–5), Moderate (5–6), Strong (6–7), Major (≥7)
- `Is_Major` – major earthquake (≥4.5)  
- `Is_Destructive` – destructive earthquake (≥5.0)

### 6. Distance Calculation

- `haversine_distance()` – compute great-circle distance (km)

### 7. Data Scaling / Normalization

Used during model training:

- `StandardScaler()` – standardize features (mean=0, std=1)
- Used for coordinate-based clustering  
- Used for regression model outputs

### 8. Data Sorting & Differencing

- `df.sort_values('Time').reset_index(drop=True)` – sort data chronologically  
- `df['Time'].diff()` – compute time differences

### 9. Missing Value Handling

- `dropna()` – remove rows containing NaN (after feature generation)  
- Rolling statistics with `min_periods=1` – allow partial windows

### 10. Encoding Transformation

- `class_weight='balanced'` – handle imbalance (classification models)  
- `LabelEncoder` – used when required by specific models


### 11. Time-Series Specific Processing

- `TimeSeriesSplit()` – time-series cross-validation  
- Preserve chronological order during training/validation  




### Why Feature Engineering Matters
- Improves model expressiveness  
- Reduces noise sensitivity  
- Allows models to detect spatial/temporal seismic patterns  
- Improves interpretability (especially with Random Forest importance scores)  

---

## **2. Multi-Model Comparison** 

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

### Regression Metrics (5)
**Primary Metrics: R² Score, MAE**

| No. | Metric | Function | Description |
|-----|--------|-----------|-------------|
| 1 | MAE | `mean_absolute_error` | Average absolute error; interpretable |
| 2 | MSE | `mean_squared_error` | Penalizes large errors heavily |
| 3 | RMSE | `mean_squared_error` (sqrt) | Root MSE, same unit as target |
| 4 | R² Score | `r2_score` | Variance explained by the model |
| 5 | MAPE (optional) | custom | Mean Absolute Percentage Error |

### Clustering Metrics (3)
**Primary Metrics: Silhouette Score, Davies–Bouldin Index (DBI)**

| No. | Metric | Function | Description |
|-----|--------|-----------|-------------|
| 1 | Silhouette Score | `silhouette_score` | Measures cluster separation (higher is better) |
| 2 | Davies–Bouldin Index | `davies_bouldin_score` | Measures cluster similarity (lower is better) |
| 3 | Calinski–Harabasz Index | custom | Measures cluster dispersion (higher is better) |

---
## **3.Hyperparameter Tuning**
This project applies a unified and systematic hyperparameter tuning strategy across all classification, regression, and clustering models. The goal is to ensure fairness, stability, and optimal performance for each model family.

### 1. Search Algorithms
Two search strategies are used depending on model complexity:

- **GridSearchCV** — exhaustive search for small/medium spaces  
- **RandomizedSearchCV** — efficient search for large or heavy models  

All tuning procedures use **5-fold cross-validation** to avoid overfitting and ensure stable model comparison.


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

---
## **Visualization**
The project includes a comprehensive visualization system designed to analyze the earthquake dataset, evaluate machine-learning models, and visualize clustering structures.
All plots are implemented in Matplotlib and Seaborn, and organized into four major components:
- Exploratory Data Analysis 
- Classification Model Visualization
- Regression Model Visualization
- Clustering Visualization
- Global Summary Dashboard  
### 1. Exploratory Data Analysis   
`Visualizer.perform_eda()` generates a 3×3 dashboard that visually explains the core structure of the dataset.

| Plot Name | Purpose |
|----------|----------|
| Magnitude Distribution | Shows the distribution of earthquake magnitudes |
| Depth Distribution | Visualizes depth variation across events |
| Spatial Distribution (Lat–Lon) | Displays geographic distribution of earthquakes |
| Daily Earthquake Frequency | Reveals daily temporal patterns |
| Monthly Distribution | Shows seasonal/monthly activity patterns |
| Hourly Distribution | Displays earthquake frequency by hour |
| Magnitude Class Distribution | Counts events in custom magnitude categories |
| Correlation Heatmap | Shows correlations between numerical features |
| Summary Statistics Panel | Displays key descriptive statistics |

**Example**
```
python
from visualizer import Visualizer

# Generate full 3×3 EDA dashboard
Visualizer.perform_eda(df)
```

### 2. Classification Model Visualization
The classification module compares 12 machine-learning classifiers using F1-Score and AUC.

| Plot Name | Purpose |
|-----------|----------|
| F1-Score Performance | Compares classifiers using F1-Score |
| F1-Score Ranking | Ranks all classification models by F1-Score |
| AUC Ranking | Ranks classification models using AUC scores |
| Feature Importance | Shows most influential predictors (tree-based models) |
| F1 vs AUC Scatter Plot | Two-metric comparison of classifier performance |

**Example**
```
python
from visualizer import Visualizer

# Compare classification models using F1 or AUC
Visualizer.plot_model_performance(classification_results, metric="F1_Score")

# Optional: feature importance for the best model
Visualizer.plot_feature_importance(best_classifier, feature_names)
```

### 3. Regression Model Visualization
The regression module compares 12 regression models using R² and MAE 

| Plot Name | Purpose |
|-----------|----------|
| R² Score Performance | Compares regressors using R² Score |
| R² Ranking | Ranks all regression models by R² |
| MAE Ranking | Ranks regressors by Mean Absolute Error |
| Feature Importance | Highlights key predictors for tree-based regressors |
| R² vs MAE Scatter Plot | Compares models on accuracy vs error trade-offs |

**Example**
```
python
from visualizer import Visualizer

# Compare regression models using R² or MAE
Visualizer.plot_model_performance(regression_results, metric="R2_Score")

# Optional: feature importance (tree-based regressors)
Visualizer.plot_feature_importance(best_regressor, feature_names)
```

### 4.Clustering Visualization
This module evaluates both KMeans and DBSCAN, and visualizes spatial clustering of earthquakes.### 4. Clustering Visualization

| Plot Name | Purpose |
|-----------|----------|
| Elbow Method (KMeans) | Helps determine the optimal number of clusters (K) |
| Silhouette Score Curve | Validates cluster structure quality |
| DBSCAN Cluster Map | Shows density-based clustering and noise points |
| KMeans Cluster Map | Visualizes spatial clusters and centroids |

**Example**
```
python
from visualizer import Visualizer

# KMeans evaluation
Visualizer.plot_kmeans_metrics(k_values, inertia_list, silhouette_list)

# Spatial cluster visualization
Visualizer.plot_clustering_results(df, labels_kmeans, labels_dbscan)

```
### 5.Final Summary Dashboard
At the end of the pipeline, a multi-panel summary figure is generated:

| Plot Name | Purpose |
|-----------|----------|
| Combined Performance Summary | Overview of classification, regression, and clustering results |
| Classification Summary Panel | Shows top-performing classification models |
| Regression Summary Panel | Displays best regression models |
| Clustering Summary Panel | Shows quality metrics of clustering algorithms |
| Text Summary Box | Final dataset + method summary printed directly in the figure |

**Example**
```
python
from visualizer import Visualizer

# Generate the full summary figure
Visualizer.plot_final_summary(
    classification_results,
    regression_results,
    clustering_metrics
)
```

---
## **Conclusion**
### Summary of the Earthquake Analysis Project

This project performed end-to-end analysis of earthquake data, including data cleaning, feature engineering, EDA, multi-model comparison (classification, regression), and spatial clustering.

**Classification Model:** To predict "major earthquakes" (magnitude ≥4.5), Random Forest achieved the best performance (F1=0.678, AUC=0.649).

**Regression Model:** To predict earthquake magnitude, Gradient Boosting performed best (R²=0.497, MAE≈0.5).

**Clustering Model:** Spatial analysis (DBSCAN/KMeans) identified 5–8 seismic clusters, with KMeans (K=2) showing clearer spatial patterns.

---
### Core Findings
1. **Temporal Patterns:** Earthquakes exhibited distinct daily and monthly patterns, with certain hours and months experiencing higher frequencies.
2. **Spatial Clustering:** Significant spatial clustering was observed, indicating regions prone to seismic activity, which could be crucial for targeted monitoring and early warning systems.
3. **Feature Importance:** Features such as spatial clusters, rolling mean magnitudes, and lagged magnitudes were identified as highly influential in predicting earthquake occurrences and magnitudes.
4. **Model Performance:** Random Forest and XGBoost emerged as top performers across both classification and regression tasks, demonstrating robustness in handling the complexity of earthquake data.
---
### Limitations of the Earthquake Analysis Project
1. **Data Constraints:**
   - The dataset size (1,308 records) is relatively small, limiting the generalization ability of complex models.
   - Missing features (e.g., tectonic plate boundaries, historical seismic activity) reduce prediction accuracy.

2. **Model Limitations:**
   - The best R² (0.497) for magnitude prediction is low—earthquake magnitude is inherently stochastic, and current features cannot fully capture its variability.
   - Classification models struggle with imbalanced data (few major earthquakes), leading to suboptimal recall.

3. **Methodological Gaps:**
   - Time-series splitting (TSCV) was used, but no external validation dataset was available to test real-world performance.
   - Spatial clustering did not integrate geological context (e.g., fault lines), so cluster interpretation is purely data-driven.
     
---
### Reflection on what the team learned
1. Technical Skills Development
- Data Engineering and Feature Engineering
  - Complex Data Processing: Learned to handle diverse data formats including temporal, spatial, and categorical earthquake data, requiring specialized cleaning and transformation techniques
  - Advanced Feature Creation: Gained expertise in creating sophisticated temporal features (lag variables, rolling statistics) and spatial features (grid-based, clustering-based) that significantly improved model performance
  - Quality Assurance: Developed robust data validation pipelines with automatic quality checks for longitude/latitude ranges, depth positivity, and magnitude validity
- Machine Learning Implementation
  - Model Diversity: Experienced implementing and comparing 32 different machine learning models across classification, regression, and clustering tasks
  - Time-Series Considerations: Learned to apply TimeSeriesSplit cross-validation to prevent data leakage in sequential earthquake data
  - Hyperparameter Tuning: Gained practical experience with RandomizedSearchCV for efficient hyperparameter optimization across multiple algorithms
  - Evaluation Metrics: Deepened understanding of appropriate metric selection (F1 for classification, R² for regression, silhouette for clustering) based on problem characteristics
- Visualization and Communication
  - Multi-dimensional Visualization: Developed skills in creating comprehensive dashboards that communicate complex seismic patterns effectively
  - Storytelling with Data: Learned to present technical findings in accessible formats for both technical and non-technical stakeholders
  - Model Interpretation: Gained experience in feature importance analysis and result explanation
2. Domain Knowledge Enhancement
- Seismic Pattern Recognition
  - Spatial Clustering: Discovered that earthquake events naturally cluster in specific geographical regions, with DBSCAN proving effective for identifying density-based patterns
  - Temporal Patterns: Observed both short-term (daily/hourly) and long-term (seasonal) patterns in earthquake occurrences
  - Magnitude-Depth Relationships: Found complex relationships between earthquake magnitude and depth that vary across geographical regions
- Predictive Challenges
  - Imbalanced Data Handling: Addressed class imbalance in earthquake prediction (most events are minor, few are major)
  - Feature Importance Insights: Learned that spatial features (cluster assignments, geographical coordinates) and temporal features (rolling statistics, lag variables) are most predictive
  - Model Selection Criteria: Discovered that ensemble methods (XGBoost, Random Forest) consistently outperform traditional algorithms for this domain
3. Collaborative Development Insights
- Workflow Optimization
  - Modular Design: Appreciated the value of creating reusable components (DataProcessor, Visualizer, ModelEvaluator classes)
  - Version Control: Learned effective Git workflows for collaborative model development
  - Documentation Practices: Recognized the importance of comprehensive documentation for reproducibility
- Problem-Solving Strategies
  - Iterative Development: Adopted an iterative approach, starting with simple models and progressively adding complexity
  - Cross-validation: Learned the importance of proper validation strategies for time-series data
  - Error Analysis: Developed systematic approaches to analyzing model failures and identifying improvement opportunities
### Possible future improvements
1. Data Level
- Increase data volume: Supplement more historical data from public data sources (such as the USGS earthquake database)
- Simple feature enhancement:
Calculate time differences and distance differences between adjacent earthquakes
Add simple geographic features: elevation, distance to the nearest city
2. Model Level
- Optimize existing best models: Fine-tune parameters for the top 2-3 performing models
- Model ensemble attempts: Try simple voting or averaging methods
- Add model explanation: Use SHAP or LIME to explain the predictions of 1-2 key models
3. Technical Deepening
- Deep learning experiments: Use simple LSTM or 1D CNN to handle time series
- Automated feature engineering: Try using FeatureTools for automated feature generation
- Model deployment practice: Package the best model into a simple web API using Flask
4. Application Expansion
- Regional segmentation analysis: Select 1-2 key regions for more detailed analysis
- Exploration of early warning time windows: Study prediction performance for different time windows (1 day/3 days/7 days)
- Influencing factor analysis: Analyze the relationship between earthquake frequency and season/time
 

