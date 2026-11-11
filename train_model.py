
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load dataset
data = pd.read_csv(
    "SMSSpamCollection.txt",
    sep="\t",
    names=["label", "message"]
)

# Step 2: Separate input and output
X = data["message"]
y = data["label"]

# Step 3: Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 4: Create Machine Learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

# Step 5: Train the model
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Check accuracy
accuracy = accuracy_score(y_test, y_pred)

print("MODEL TRAINED SUCCESSFULLY!")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))

# Step 8: Save trained model
joblib.dump(model, "spam_model.pkl")

print("\nModel saved successfully!")

