import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 1. Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# 2. Encode labels: ham = 0, spam = 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# 4. Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 6. Evaluate model
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 7. Predict new SMS
def predict_sms(text):
    text_vec = vectorizer.transform([text])
    pred = model.predict(text_vec)[0]
    return "Spam" if pred == 1 else "Ham"

# Example
print(predict_sms("Congratulations! You've won a free ticket. Call now!"))
