# 🚀 BoostArena: Dynamic Tree-by-Tree Visualizer
<img width="2760" height="1400" alt="boostarena_project_banner" src="https://github.com/user-attachments/assets/671ec432-785a-4f1d-98ad-5678bf68f2a2" />

### 🥊 The Ultimate Ensemble Learning Showdown.

BoostArena is an interactive Machine Learning dashboard built with **Streamlit** and **Plotly** that visualizes the tree-by-tree optimization paths of **AdaBoost**, **Gradient Boosting**, and **XGBoost** in real-time. 

Instead of showing static final metrics, this application tracks validation log-loss iteration-by-iteration. Upload any classification dataset—like the provided Travel Tour package dataset—to watch how sequential estimators adjust weights and minimize errors live.

---

## 🌟 Key Features
* **Dynamic Data Ingestion:** Upload any custom classification CSV file directly through the UI.
* **Smart Preprocessing Pipeline:** Automatically detects column data types, handles missing values (using median/mode imputation), strips high-cardinality ID columns, and applies one-hot encoding.
* **Tree-by-Tree Analysis:** Captures and plots validation loss after *each* consecutive tree is added to the ensemble using `staged_predict_proba`.
* **Hyperparameter Playgrounds:** Interactive sidebar controls to change the number of estimators (trees) and learning rates on the fly.

---------------------------------------------------------------------------------------------------------------------------------------------------
<img width="1260" height="750" alt="Screenshot 2026-06-24 121826" src="https://github.com/user-attachments/assets/18af1da4-f000-4e31-85b7-23803449a062" />
<img width="1277" height="400" alt="Screenshot 2026-06-24 121814" src="https://github.com/user-attachments/assets/8435caa1-616d-4cb9-a456-116a4b937dcc" />
<img width="1260" height="675" alt="Screenshot 2026-06-24 121803" src="https://github.com/user-attachments/assets/cd166efd-1d4e-4ad2-bc24-d1c538e90658" />
<img width="1861" height="695" alt="Screenshot 2026-06-24 121747" src="https://github.com/user-attachments/assets/fd145b84-1b1c-4aa6-bae2-1696b0d86b4c" />
<img width="1852" height="422" alt="Screenshot 2026-06-24 121731" src="https://github.com/user-attachments/assets/fd2880a7-b99e-4a10-8922-5c2ac1a873d8" />
<img width="446" height="552" alt="Screenshot 2026-06-24 121840" src="https://github.com/user-attachments/assets/ae4601b7-d262-43b4-b15d-c559cacb088a" />

---------------------------------------------------------------------------------------------------------------------------------------------------
## 📊 Sample Dataset Included
The project is optimized out-of-the-box to handle **`Travel (1).csv`**.
* **Target Variable:** `ProdTaken` (Whether a customer purchased the holiday package).
* **Predictors:** Features like `Age`, `MonthlyIncome`, `ProductPitched`, `Designation`, and `MaritalStatus`.

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/boost-arena.git](https://github.com/YOUR_USERNAME/boost-arena.git)
cd boost-arena
