import pandas as pd

file_path = "FinancialPhraseBank/Sentences_AllAgree.txt"

data = []
with open(file_path, "r", encoding="latin-1") as f:
    for line in f:
       
        label, sentence = line.strip().split("@")
        data.append([sentence.strip(), label.strip()])

df = pd.DataFrame(data, columns=["sentence", "sentiment"])
print(df.head())
print(df['sentiment'].value_counts())


import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower() 
    text = re.sub(r'\d+', '', text)  
    text = text.translate(str.maketrans('', '', string.punctuation)) 
    text = " ".join([w for w in text.split() if w not in stop_words])  
    return text

df['clean_text'] = df['sentence'].apply(clean_text)
print(df[['sentence', 'clean_text']].head())


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['clean_text']).toarray()
y = df['sentiment']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
print("\n Confusion Matrix:\n", cm)


print("\n Sentiment Distribution:\n")
print(df['sentiment'].value_counts())


from collections import Counter

def get_top_n_words(corpus, n=10):
    all_words = ' '.join(corpus).split()
    counter = Counter(all_words)
    return counter.most_common(n)

print("\n Top Positive Words:", get_top_n_words(df[df['sentiment']=='positive']['clean_text']))
print("\n Top Negative Words:", get_top_n_words(df[df['sentiment']=='negative']['clean_text']))
print("\n Top Neutral Words:", get_top_n_words(df[df['sentiment']=='neutral']['clean_text']))

