
import pandas as pd
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Download NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load the dataset
df = pd.read_csv('spam.csv', encoding='ISO-8859-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# Map labels to binary values
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    stop_words = stopwords.words('english')
    return ' '.join([word for word in words if word not in stop_words])

# Apply cleaning
df['cleaned'] = df['message'].apply(clean_text)

# Vectorize the text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['cleaned'])

# Labels
y = df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Output results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Predict on custom message
example = ["Congratulations! You've won a free ticket. Call now!"]
example_cleaned = [clean_text(msg) for msg in example]
example_vectorized = vectorizer.transform(example_cleaned)
prediction = model.predict(example_vectorized)

print("Example prediction:", "Spam" if prediction[0] else "Ham")
