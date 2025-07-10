# email-spam-detection
A machine learning model that detects whether an email is spam or not using NLP and scikit-learn.

# 📧 Email Spam Detection using Machine Learning

This project detects whether an email is **spam** or **not spam** using a machine learning model trained on labeled email text data. The model is built using Python, scikit-learn, and NLP techniques in Google Colab.

---

## 📌 Features

- Preprocessing of email text using tokenization, stopword removal, and TF-IDF
- Training with models like Naive Bayes / Logistic Regression
- Model evaluation using accuracy, confusion matrix, and classification report
- Built and tested in Google Colab

---

## 🧰 Technologies Used

- Python 3
- scikit-learn
- Pandas, NumPy
- NLTK
- Google Colab
- Matplotlib / Seaborn (for visualizations)

---

## ⚙️ How to Run

1. Open `Email_Spam_Detection.ipynb` in Google Colab.
2. Upload the dataset (CSV format with labeled spam/ham emails).
3. Run all cells:
   - Preprocess the text data
   - Train ML models
   - Evaluate accuracy and visualize results

---


## 📂 Dataset

Used publicly available spam email dataset from [Kaggle](https://www.kaggle.com/purusinghvi/email-spam-classification-dataset) or UCI ML repository.

---

## 🔮 Future Work

- Add deep learning (LSTM/BERT) for more accurate classification
- Deploy as a web application with a simple user interface
- Add real-time email detection using APIs

---

## 📄 License

For educational use only. Dataset credits belong to their original authors.
