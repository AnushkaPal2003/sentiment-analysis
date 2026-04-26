# 🎬 Sentiment Analysis App

This project predicts whether a movie review is **positive or negative** using machine learning.

## 📌 What I did
- Cleaned text data using basic NLP preprocessing
- Converted text into numerical features using TF-IDF
- Trained multiple models (Logistic Regression, Naive Bayes, SVM)
- Used MLflow to compare models and select the best one
- Built a simple UI using Streamlit
- Dockerized the app and deployed it on Azure

## 🚀 How to run locally

1. Clone the repo  
2. Install dependencies:
   pip install -r requirements.txt  

3. Train the model:
   python train.py  

4. Run the app:
   streamlit run streamlit_app.py  

## 📸 Screenshots

### 🔹 App Interface
User enters a movie review and gets sentiment prediction.

![App Screenshot](Images\app.png)


### 🔹 Deployment Status
Azure deployment success screen.

![Deployment Screenshot](Images\Deployment.png)



## ☁️ Deployment
The app was deployed using Docker on Azure.

## 🧠 Tech used
- Python
- Scikit-learn
- MLflow
- Streamlit
- Docker
- Azure