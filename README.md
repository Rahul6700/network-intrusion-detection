# 🧠 NSL-KDD Intrusion Detection using Machine Learning

This project implements **multiple machine learning models** (Decision Tree, Random Forest, and Gaussian Naive Bayes) to detect different types of network intrusions using the **NSL-KDD dataset**.

The code automatically downloads the dataset using **KaggleHub**, preprocesses it, encodes categorical variables, scales numeric features, trains models, and prints evaluation metrics.

The code was originally developed and executed in **Google Colab / Jupyter Notebook**, meaning there is **no virtual environment (venv)** or dependency management included in the repository.  
If you clone this repository, you’ll need to **manually install the required dependencies** before running the code (use a venv :)).


---

## 📚 Dataset

The project uses the [NSL-KDD dataset](https://www.kaggle.com/datasets/hassan06/nslkdd), which is an improved version of the classic KDD Cup 1999 dataset.  
It contains records of network connections labeled as **normal** or as one of several types of **attacks**.

---

## ⚙️ Features of the Project

- Automatic dataset download using `kagglehub`
- One-hot encoding of categorical features
- Data scaling using `StandardScaler`
- Attack label mapping into 5 broad categories:
  - **DoS**
  - **Probe**
  - **R2L**
  - **U2R**
  - **Normal**
- Training and evaluation of three models:
  - Decision Tree Classifier  
  - Random Forest Classifier  
  - Gaussian Naive Bayes

---

## 🧩 Project Structure

### 🧠 Models and Evaluation

The code trains and evaluates:
1. **Decision Tree**
2. **Random Forest**
3. **Gaussian Naive Bayes**

Each model outputs:
- Accuracy score
- Full classification report (Precision, Recall, F1-score)

---
