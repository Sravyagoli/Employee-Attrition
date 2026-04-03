# 👥 Employee Attrition Prediction

A machine learning project that analyzes and predicts employee attrition using the IBM HR Analytics dataset. The goal is to identify key factors that contribute to employee turnover and build a classification model to predict whether an employee is likely to leave the company.

---

## 📊 Dataset

**Source:** IBM HR Analytics Employee Attrition & Performance Dataset (`WA_Fn-UseC_-HR-Employee-Attrition.csv`)

- **Rows:** 1,470 employee records
- **Columns:** 35 features covering demographics, job details, satisfaction scores, and compensation
- **Target Variable:** `Attrition` (Yes / No)
- **Class Distribution:** 1,233 stayed (84%) vs. 237 left (16%)
- **Missing Values:** None ✅

---

## 🔄 Project Workflow

### 1. 🔍 Exploratory Data Analysis (EDA)
- Inspected data types, shapes, and summary statistics
- Checked for missing/null values (dataset is clean)
- Visualized attrition counts and attrition by age group
- Examined unique value distributions for all categorical columns

### 2. 🛠️ Data Preprocessing
- Dropped columns with zero variance or no predictive value: `Over18`, `EmployeeNumber`, `StandardHours`, `EmployeeCount`
- Computed a correlation heatmap to understand feature relationships
- Encoded all categorical features using `LabelEncoder`
- Renamed the `Age` column to `Age_Years` for clarity

### 3. 🌲 Model Training
- **Algorithm:** Random Forest Classifier
- **Train/Test Split:** 75% training / 25% testing
- **Hyperparameters:** `n_estimators=10`, `criterion='entropy'`, `random_state=0`

### 4. 📈 Model Evaluation
- Evaluated using a confusion matrix and overall accuracy

| Metric | Value |
|---|---|
| 🏋️ Training Accuracy | **97.91%** |
| 🧪 Testing Accuracy | **86.41%** |

**🔢 Confusion Matrix (Test Set):**

|  | Predicted No | Predicted Yes |
|---|---|---|
| **Actual No** | 309 | 1 |
| **Actual Yes** | 49 | 9 |

---

## 🧰 Technologies Used

- 🐍 **Python 3**
- 🐼 **pandas** — data manipulation
- 🔢 **numpy** — numerical operations
- 📉 **matplotlib / seaborn** — data visualization
- 🤖 **scikit-learn** — machine learning (LabelEncoder, train_test_split, RandomForestClassifier, confusion_matrix)

---

## 📁 Files

```
├── WA_Fn-UseC_-HR-Employee-Attrition.csv   # Raw dataset
├── Employee_Attrition.ipynb                 # Main Jupyter notebook
└── README.md                                # Project documentation
```

---

## 💡 Key Findings

- Only **16.1%** of employees left, making this a class-imbalanced problem ⚠️
- The Random Forest model achieves strong overall accuracy but has low recall for the minority class (employees who left), which is common with imbalanced datasets.
- 🚀 Potential improvements include using SMOTE for oversampling, tuning `n_estimators`, or trying other algorithms like XGBoost or Logistic Regression with class weighting.

---

## ▶️ How to Run

1. Clone or download the repository.
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Open `Employee_Attrition.ipynb` in Jupyter Notebook or Google Colab.
4. Place `WA_Fn-UseC_-HR-Employee-Attrition.csv` in the same directory.
5. Run all cells sequentially. 🎉
