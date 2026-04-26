# Sentiment Analysis App 🎬

This is an end-to-end machine learning project where I built a model to classify movie reviews as positive or negative.


## What I did

* Cleaned and preprocessed text data
* Converted text into features using TF-IDF
* Trained multiple models (Logistic Regression, Naive Bayes, SVM)
* Compared their performance and selected the best one
* Tracked experiments using MLflow
* Built a simple web app using Streamlit
* Dockerized the application
* Deployed it on Azure earlier (currently inactive due to subscription limits)


## Model Performance

* Logistic Regression: ~88% accuracy (best)
* SVM: ~87%
* Naive Bayes: ~85%


## How to run the project

### 1. Install dependencies

pip install -r requirements.txt

### 2. Train the model

python train.py

### 3. Run the app

streamlit run streamlit_app.py

## Docker 

Build image:
docker build -t sentiment-app .

Run container:
docker run -p 8501:8501 sentiment-app


## Demo

Below are screenshots of the deployed app and prediction results:

### App Prediction

(<Images-azure/app image.png>)

### Azure Deployment

(<Images-azure/Deployment image.png>)


## About

This project helped me understand the full workflow of an ML project — from training to deployment.
