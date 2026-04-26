import os
import mlflow
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

from preprocess import clean_text

os.makedirs("models", exist_ok=True)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Sentiment Analysis")

df = pd.read_csv("IMDB Dataset.csv")
df = df.sample(15000, random_state=42)

df["clean_review"] = df["review"].apply(clean_text)
df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})

X = df["clean_review"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC()
}

best_accuracy = 0
best_model_name = ""
best_pipeline = None

for name, model in models.items():

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("model", model)
    ])

    with mlflow.start_run(run_name=name):

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)

        mlflow.log_param("model_name", name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", report["weighted avg"]["f1-score"])

        mlflow.sklearn.log_model(pipeline, "model")

        print(name, accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_pipeline = pipeline

joblib.dump(best_pipeline, "models/model.pkl")

print("Best Model:", best_model_name)
print("Best Accuracy:", best_accuracy)