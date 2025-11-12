# 🧠 Breast Cancer Detection Using Machine and Deep Learning

## 📄 Overview
This project focuses on detecting breast cancer using both **Machine Learning (ML)** and **Deep Learning (DL)** techniques. The goal is to analyze and predict whether a tumor is **benign** or **malignant** using the **Breast Cancer dataset** integrated from the **Scikit-learn library**.

The notebook demonstrates the complete process — from data preprocessing to model evaluation — implemented on **Google Colab** for ease of computation and visualization.

---

## 🧰 Tools and Technologies
- **Language:** Python  
- **Platform:** Google Colab  
- **Libraries Used:**
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `tensorflow` / `keras` (for deep learning models)

---

## 📊 Dataset
- **Source:** Breast Cancer dataset from `sklearn.datasets`
- **Description:**
  - **Features:** 30 numeric features representing characteristics of cell nuclei  
  - **Target:** Binary classification — *0 for malignant*, *1 for benign*  
- **Shape:** 569 samples × 30 features

---

## ⚙️ Project Workflow
1. **Importing Libraries and Dataset**
   - Load the built-in Breast Cancer dataset from Scikit-learn.
2. **Exploratory Data Analysis (EDA)**
   - Check for missing values, data distribution, and correlation heatmap.
3. **Data Preprocessing**
   - Standardize data using `StandardScaler`.
   - Split dataset into training and testing sets.
4. **Model Building**
   - **Machine Learning Models:** Logistic Regression, Random Forest, SVM, Decision Tree, etc.  
   - **Deep Learning Model:** Artificial Neural Network (ANN) using Keras/TensorFlow.
5. **Model Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
6. **Result Comparison**
   - Compare ML and DL model performances visually using bar plots.

---

## 🧪 Results
- ML models achieved around **95–98% accuracy**, depending on the algorithm.  
- The **ANN model** provided comparable or slightly better results with strong generalization.  
- Deep learning showed potential for handling more complex or large-scale medical datasets.

---

## 🚀 How to Run
1. Open **Google Colab** or **Jupyter Notebook**.
2. Upload the `.ipynb` file.
3. Run all cells sequentially.
4. Ensure the required libraries are installed:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
5. View results in the output cells and performance visualizations.

## Future Improvements 
- Implement CNN or hybrid models for image-based breast cancer datasets.

- Add hyperparameter tuning using GridSearchCV or Keras Tuner.

- Deploy the model using Flask or Streamlit for real-time prediction.

