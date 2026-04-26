import os
import mlflow
import pandas as pd
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report   

from preprocess import clean_text

# Create model folder
os.makedirs('models', exist_ok=True)

# Load the dataset
print("Loaded dataset")
df=pd.read_csv('IMDB Dataset.csv')
df = df.sample(15000, random_state=42)
print("Cleaning text...")
df['clean_review']=df['review'].apply(clean_text)
df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})

x=df['clean_review']
y=df['label']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
# Vectorize the text data
print("Vectorizing...")
vectorizer=TfidfVectorizer(max_features=5000)
print("Splitting...")
x_train_vec=vectorizer.fit_transform(x_train)
x_test_vec=vectorizer.transform(x_test)

models={
    'Logistic Regression':LogisticRegression(max_iter=1000),
    'Naive Byes': MultinomialNB(),
    'SVM': LinearSVC()
}

best_accuracy=0
best_model=None
best_model_name=''
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment('sentiment Analysis')
for name,model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(x_train_vec,y_train)
        pred=model.predict(x_test_vec)

        accuracy=accuracy_score(y_test,pred)
    
        mlflow.log_param('Model_Name',name)
        mlflow.log_metric('Accuracy',accuracy)

        mlflow.sklearn.log_model(model, 'model')

        print(f'{name} Accuracy: {accuracy}')

        if accuracy>best_accuracy:
            best_model=model
            best_model_name=name
            best_accuracy=accuracy

# Save best model
joblib.dump(best_model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\nBest Model:", best_model_name)
print("Best Accuracy:", best_accuracy)