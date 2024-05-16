import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('stopwords')
nltk.download('wordnet')


lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Ensure text is a string
    if not isinstance(text, str):
        return ""  
    
    text = text.lower()  # Lowercase text
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)  # Remove non-letters
    text = text.strip()  # Remove whitespace
    tokens = text.split()  # Tokenize text
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]  # Lemmatize and remove stopwords
    return ' '.join(tokens)

# Load your CSV file
data = pd.read_csv('/content/drive/MyDrive/Suicide_Detection.csv')
print(data.columns)


data['Clean_Tweet'] = data['text'].apply(clean_text)


print(data[['text', 'Clean_Tweet']].head())



data.to_csv('cleaned_tweets.csv', index=False)


data[['Clean_Tweet']].to_csv('cleaned_tweets.csv', index=False)


from sklearn.feature_extraction.text import TfidfVectorizer


vectorizer = TfidfVectorizer(max_features=10000)  
tfidf_features = vectorizer.fit_transform(data['Clean_Tweet'])


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  
    'solver': ['liblinear', 'lbfgs'],  
    'class_weight': [None, 'balanced'] 
}


grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(tfidf_features,data['class'].apply(lambda x: "Potential Suicide post" if x == "suicide" else "Not Suicide post"))


print("Best parameters:", grid_search.best_params_)
print("Best score: {:.2f}".format(grid_search.best_score_))

from sklearn.metrics import classification_report, confusion_matrix


best_model = grid_search.best_estimator_
predictions = best_model.predict(tfidf_features)


print(classification_report(data['class'].apply(lambda x: "Potential Suicide post" if x == "suicide" else "Not Suicide post"), predictions))
print("Confusion Matrix:\n", confusion_matrix(data['class'].apply(lambda x: "Potential Suicide post" if x == "suicide" else "Not Suicide post"), predictions))


import joblib

joblib.dump(best_model, 'suicidal_tendency_detector.pkl')
