import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

# 🔹 Step 1: Load cleaned dataset
df = pd.read_csv(r"D:\sriram\Projects\Fake news detection\data\cleaned_fake_news_dataset.csv")

# 🔹 Step 2: Split data
X = df['clean_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 🔹 Step 3: Convert text → TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 🔹 Step 4: Train Logistic Regression model
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_tfidf, y_train)

# 🔹 Step 5: Evaluate model
y_pred = model.predict(X_test_tfidf)

print("✅ Model Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Real","Fake"]))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 🔹 Step 6: Save model + vectorizer
os.makedirs(r"D:\sriram\Projects\Fake news detection\models", exist_ok=True)

joblib.dump(model, r"D:\sriram\Projects\Fake news detection\models\fake_news_model.pkl")
joblib.dump(vectorizer, r"D:\sriram\Projects\Fake news detection\models\tfidf_vectorizer.pkl")

print("\n✅ Model & Vectorizer saved successfully!")
