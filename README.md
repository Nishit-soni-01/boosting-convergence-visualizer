# 🚀 BoostArena: Dynamic Tree-by-Tree Visualizer
### 🥊 The Ultimate Ensemble Learning Showdown.

BoostArena is an interactive Machine Learning dashboard built with **Streamlit** and **Plotly** that visualizes the tree-by-tree optimization paths of **AdaBoost**, **Gradient Boosting**, and **XGBoost** in real-time. 

Instead of showing static final metrics, this application tracks validation log-loss iteration-by-iteration. Upload any classification dataset—like the provided Travel Tour package dataset—to watch how sequential estimators adjust weights and minimize errors live.

---

## 🌟 Key Features
* **Dynamic Data Ingestion:** Upload any custom classification CSV file directly through the UI.
* **Smart Preprocessing Pipeline:** Automatically detects column data types, handles missing values (using median/mode imputation), strips high-cardinality ID columns, and applies one-hot encoding.
* **Tree-by-Tree Analysis:** Captures and plots validation loss after *each* consecutive tree is added to the ensemble using `staged_predict_proba`.
* **Hyperparameter Playgrounds:** Interactive sidebar controls to change the number of estimators (trees) and learning rates on the fly.

---

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
