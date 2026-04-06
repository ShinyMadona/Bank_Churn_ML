https://bankchurnml-plbplrsg2zhasqetwze8de.streamlit.app

# Bank Customer Churn Prediction

A machine learning project that predicts whether a bank customer is likely to **churn** based on demographic, financial, and account-related features.  
The project focuses on identifying churn patterns and evaluating multiple classification models using robust performance metrics.

---

## Tech Stack
- Python
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn

---

## Dataset
- Bank customer churn dataset
- Structured tabular data containing customer demographics and account information
- Target variable:
  - Churn (Exited / Not Exited)

---

## Approach
- Performed exploratory data analysis (EDA) to understand customer behavior
- Handled feature encoding and scaling
- Trained multiple classification models including:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest
- Compared models using:
  - Accuracy
  - Confusion Matrix
  - Classification Report
  - ROC-AUC Curve

---

## Results
- Random Forest and XG Boost achieved best performance
- ROC curve shows strong class separation capability
- Feature importance analysis highlights key drivers of customer churn

Key result artifact:
- `ROC_Curve.png`

---

## Project Structure
bank-churn-prediction/
├── BANK_CHURN_DATA.ipynb
├── ROC_Curve.png
├── README.md
├── requirements.txt


---

## How to Run
1. Clone the repository  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Open BANK_CHURN_DATA.ipynb in Jupyter Notebook or Google Colab
4. Run all cells sequentially

## Business Insight

The model helps identify customers at high risk of churn, enabling banks to take proactive retention measures such as targeted offers or personalized engagement strategies.

## Future Improvements

1. Handle class imbalance using advanced techniques
2. Perform hyperparameter tuning for all models
3. Add SHAP-based model explainability
