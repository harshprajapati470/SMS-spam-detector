import pandas as pd
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

nltk.download('stopwords')
from nltk.corpus import stopwords

# Load data
df = pd.read_csv('spam.csv', encoding='ISO-8859-1')[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Clean text
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    stop_words = stopwords.words('english')
    return ' '.join([word for word in words if word not in stop_words])

df['cleaned'] = df['message'].apply(clean_text)

# Vectorize and train model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['cleaned'])
y = df['label']

model = MultinomialNB()
model.fit(X, y)

# Save model and vectorizer
joblib.dump(model, 'spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("Model and vectorizer have been saved successfully.")