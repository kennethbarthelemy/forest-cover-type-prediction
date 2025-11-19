# Forest Cover Type Prediction using Machine Learning

[![R](https://img.shields.io/badge/R-276DC3?style=flat&logo=r&logoColor=white)](https://www.r-project.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Project-blue)](https://github.com)
[![Status](https://img.shields.io/badge/Status-Completed-success)](https://github.com)

## 📋 Project Overview

This project applies machine learning techniques to predict forest cover types in Roosevelt National Forest, Northern Colorado, using cartographic data from the U.S. Forest Service. The goal is to classify 30×30 meter land cells into one of seven forest cover types based on environmental features.

**Course**: QTM 7571 - Machine Learning Methods  
**Institution**: Babson College  
**Team**: Group 4 (Kenneth Barthelemy, Dilo Junior, Javier Martinez, Priyanka Tambe)  
**Dataset**: [Kaggle - Forest Cover Type Prediction](https://www.kaggle.com/c/forest-cover-type-prediction)

---

## 🎯 Key Results

| Model | Error Rate | Accuracy | Performance vs Baseline |
|-------|-----------|----------|------------------------|
| Baseline (Mode Prediction) | 86.5% | 13.5% | - |
| K-Nearest Neighbors (Phase 1) | 28.5% | 71.5% | 67% improvement |
| Logistic Regression (Phase 1) | 30.3% | 69.7% | 65% improvement |
| **Random Forest (Phase 2)** | **14.4%** | **85.6%** | **83% improvement** |
| **Stacking Ensemble (Phase 2)** | **14.4%** | **85.6%** | **83% improvement** |

---

## 📊 Dataset Description

- **Observations**: 15,120 training samples
- **Features**: 54 input features (12 numerical + 42 categorical)
- **Target Variable**: Cover_Type (7 classes)
- **Classes**: 
  1. Spruce/Fir
  2. Lodgepole Pine
  3. Ponderosa Pine
  4. Cottonwood/Willow
  5. Aspen
  6. Douglas-fir
  7. Krummholz

### Features

**Numerical Variables** (10):
- Elevation
- Aspect
- Slope
- Horizontal Distance to Hydrology
- Vertical Distance to Hydrology
- Horizontal Distance to Roadways
- Hillshade at 9am, Noon, 3pm
- Horizontal Distance to Fire Points

**Categorical Variables** (44):
- Wilderness Area (4 binary indicators)
- Soil Type (40 binary indicators)

---

## 🔧 Technical Approach

### Data Preprocessing
1. **Data Quality Check**: Verified no missing values across 15,120 observations
2. **Feature Consolidation**: Combined 44 binary columns into 2 factor variables
   - Wilderness Area: 4 categories → single factor
   - Soil Type: 40 categories → single factor
3. **Feature Engineering**:
   - Created `Total_Distance` (sum of all distance features)
   - Created `Slope_Elevation` interaction term
4. **Standardization**: Centered and scaled all numerical features
5. **Train-Test Split**: 60% training / 40% testing

### Models Implemented

#### Phase 1: Foundational Models
1. **K-Nearest Neighbors (KNN)**
   - K = 5 (Phase 1), K = 123 (Phase 2, using √N rule)
   - Standardized numerical features only
   - Error Rate: 28.5% → 40.4%
   
2. **Multinomial Logistic Regression**
   - Stepwise variable selection for overfitting control
   - Handles mixed data types
   - Error Rate: 30.3% → 29.3%

#### Phase 2: Advanced Models
3. **Random Forest**
   - 100 decision trees
   - Handles categorical variables natively
   - Variable importance analysis
   - **Error Rate: 14.4%**
   - **Top Predictors**: Elevation, Soil Type, Distance to Roadways

4. **Stacking Ensemble**
   - Meta-model: Neural Network (1 hidden layer, 6 nodes)
   - Base models: KNN, Logistic Regression, Random Forest
   - Combined predictions as features
   - **Error Rate: 14.4%**

---

## 📈 Model Performance Comparison

### Overall Accuracy by Cover Type

| Cover Type | KNN | Logistic Regression | Random Forest | Stacking |
|-----------|-----|---------------------|---------------|----------|
| Type 1 | 43% | 65% | 76% | 76% |
| Type 2 | 34% | 56% | 68% | 69% |
| Type 3 | 35% | 43% | 81% | 80% |
| Type 4 | 86% | 68% | 96% | 97% |
| Type 5 | 74% | 70% | 95% | 94% |
| Type 6 | 63% | 52% | 86% | 86% |
| Type 7 | 82% | 75% | 98% | 98% |

### Key Insights
- **Cover Types 4, 5, and 7** were easiest to predict (distinct environmental patterns)
- **Cover Types 1 and 2** were most challenging (overlapping characteristics)
- **Elevation** emerged as the single most important predictor
- **Ensemble methods** significantly outperformed individual models

---

## 🛠️ Technologies & Tools

- **Language**: R (version 4.x)
- **Libraries**:
  - `caret` - Machine learning workflows
  - `class` - K-Nearest Neighbors
  - `nnet` - Neural Networks & Multinomial Logistic Regression
  - `randomForest` - Random Forest implementation
  - `ggplot2` - Data visualization
  - `reshape2` - Data transformation

---

## 📂 Repository Structure
```
forest-cover-prediction/
│
├── scripts/
│   ├── phase1_knn_logistic.R          # Initial models (KNN & Logistic Regression)
│   └── phase2_advanced_models.R       # Advanced models (Random Forest & Stacking)
│
├── reports/
│   ├── Phase0_Project_Proposal.pdf
│   ├── Phase1_Initial_Models.pdf
│   └── Phase2_Final_Report.pdf
│
└── README.md
```

---

## 🚀 How to Run

### Prerequisites
```r
# Install required packages
install.packages(c("rpart", "rpart.plot", "ggplot2", "reshape2", 
                   "caret", "nnet", "randomForest", "class"))
```

### Execution
1. Download the [training dataset from Kaggle](https://www.kaggle.com/c/forest-cover-type-prediction/data)
2. Update file paths in the R scripts
3. Run Phase 1 models:
```r
   source("scripts/phase1_knn_logistic.R")
```
4. Run Phase 2 models:
```r
   source("scripts/phase2_advanced_models.R")
```

---

## 📊 Visualizations

### Cover Type Distribution
- Balanced dataset: 2,160 observations per class
- No class imbalance issues

### Feature Analysis
- **Elevation**: Strong differentiator across cover types
- **Hillshade variables**: Moderate predictive power
- **Distance features**: Right-skewed distributions (most areas near water/roads)

---

## 🎓 Learning Outcomes

1. **Model Selection**: Understanding trade-offs between interpretability and accuracy
2. **Feature Engineering**: Creating meaningful derived features improves performance
3. **Ensemble Methods**: Combining models can capture complex patterns individual models miss
4. **Overfitting Prevention**: Techniques like stepwise selection, cross-validation, and ensemble learning
5. **Real-World Applications**: Practical use in environmental science and land management

---

## 🔮 Future Improvements

- [ ] Implement cross-validation for more robust error estimates
- [ ] Explore gradient boosting methods (XGBoost, LightGBM)
- [ ] Feature selection using recursive feature elimination
- [ ] Hyperparameter tuning with grid search
- [ ] Deploy model as web application
- [ ] Test on Kaggle's test set (565,892 observations)

---

## 📚 References

- Dataset: [Kaggle Forest Cover Type Prediction](https://www.kaggle.com/c/forest-cover-type-prediction)
- Original Research: Blackard, Jock A. and Denis J. Dean. 2000. "Comparative Accuracies of Artificial Neural Networks and Discriminant Analysis in Predicting Forest Cover Types from Cartographic Variables." *Computers and Electronics in Agriculture* 24(3):131-151.

---

## 👥 Contributors

- **Kenneth Barthelemy** - [LinkedIn](your-linkedin-url)
- **Dilo Junior**
- **Javier Martinez**
- **Priyanka Tambe**

---

## 📄 License

This project was completed as part of academic coursework at Babson College.

---

## 🤝 Acknowledgments

Special thanks to the QTM 7571 course instructor and the U.S. Forest Service for providing the dataset.

---

*Last Updated: May 2025*
