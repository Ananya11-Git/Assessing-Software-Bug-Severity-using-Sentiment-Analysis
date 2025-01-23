    # -*- coding: utf-8 -*-

#code to download all pages of bug reports from the issues tab of any github repository.
#anthropic

import requests
import json
import csv
import pandas as pd
from google.colab import files

owner = 'EllanJiang' #username of the repository owner
repo = 'GameFramework' #name of repository


url = f'https://api.github.com/repos/{owner}/{repo}/issues?state=all'

# Your GitHub Personal Access Token, generate with developers tab in settings
#only generated once, so save them all somewhere else too
token = '' #insert api key here

# Set up headers with authentication
headers = {
    'Authorization': f'token {token}',
    'Accept': 'application/vnd.github.v3+json'
}

# Initialize variables for pagination
structured_data = []
page = 1
has_next_page = True

while has_next_page:
    # Make the API request
    response = requests.get(url + f'&page={page}', headers=headers)

    # Check for successful response
    if response.status_code == 200:
        issues = response.json()

        # Check if there are issues in the response
        if len(issues) == 0:
            break

        # Extract structured data from each issue
        for issue in issues:
            issue_data = {
                'number': issue['number'],
                'title': issue['title'],
                'state': issue['state'],
                'labels': [label['name'] for label in issue['labels']],
                'created_at': issue['created_at'],
                'updated_at': issue['updated_at'],
                'closed_at': issue['closed_at'] if 'closed_at' in issue else None,
                'author': issue['user']['login']
            }
            structured_data.append(issue_data)

        # Move to the next page
        page += 1

        # Check if there's another page
        has_next_page = 'next' in response.links and response.links['next']['url'] is not None
    else:
        print(f'Failed to fetch issues: {response.status_code} - {response.text}')
        break

# Create a DataFrame
df = pd.DataFrame(structured_data)

# Save the DataFrame to a CSV file
csv_file = 'game_all_bugs.csv'
df.to_csv(csv_file, index=False)

# Download the CSV file
files.download(csv_file)

!pip install pandas scikit-learn nltk imbalanced-learn

import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from textblob import TextBlob
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Load your data
data = pd.read_csv('anthropics_all_bugs.csv')
data['title']= data['title'].apply(str)

# Example data structure: data['summary'], data['priority']

# Preprocessing functions
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
  if not isinstance(text,str):
    return ''

  tokens = word_tokenize(text)
  tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalpha() and token not in stop_words]
  return ' '.join(tokens)

data['clean_title'] = data['title'].apply(preprocess)

from google.colab import files

def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity
data['sentiment'] = data['clean_title'].apply(get_sentiment)

!pip install nlpaug
import nlpaug.augmenter.word as naw
# Define the augmenter
augmenter = naw.SynonymAug(aug_src='wordnet')
def augment_text(text):
  return augmenter.augment(text)

data['title'] = data['title'].astype(str)


# Classify severity
severity = []
for i in data['sentiment']:
    if i < 0:
        severity.append('Critical')
    elif i == 0:
        severity.append('Major')
    else:
        severity.append('Minor')

data['Severity Prediction'] = severity
data.to_csv('anthlabeling.csv')
files.download('anthlabeling.csv')

X = data[['title', 'sentiment']]
y = data['Severity Prediction']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Preprocessing pipeline
vectorizer = TfidfVectorizer()

# Models
knn = KNeighborsClassifier()
svm = SVC(probability=True)
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()

# Ensemble model
ensemble = VotingClassifier(estimators=[
    ('knn', knn),
    ('svm', svm),
    ('dt', dt),
    ('rf', rf)
], voting='soft')

# Create a pipeline
pipeline = Pipeline([
    ('vectorizer', vectorizer),
    ('classifier', ensemble)
])

# Train the model
pipeline.fit(X_train['title'], y_train)

y_pred = pipeline.predict(X_test['title'])
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Save the pipeline and metrics to pickle files
with open('model_pipeline.pkl', 'wb') as file:
    pickle.dump(pipeline, file)

metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1
}

with open('metrics.pkl', 'wb') as file:
    pickle.dump(metrics, file)

!pip install gradio
import gradio as gr
import pandas as pd
import pickle
from textblob import TextBlob

# Load the trained pipeline and metrics
with open('model_pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)

with open('metrics.pkl', 'rb') as file:
    metrics = pickle.load(file)

def predict_severity(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file.name)
    df['title'] = df['title'].apply(str)  # Ensure all text data is string

    # Add sentiment score as a feature
    df['sentiment'] = df['title'].apply(lambda text: TextBlob(text).sentiment.polarity)

    # Make predictions
    predictions = pipeline.predict(df['title'])

    # Add predictions to the DataFrame
    df['predicted_severity'] = predictions

    # Save the DataFrame to a CSV file
    output_csv = "output_with_predictions.csv"
    df.to_csv(output_csv, index=False)

    image = "/content/workflow_diagram (1).png"

    return image, df, metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1'], output_csv

#reate the Gradio interface
interface = gr.Interface(
    fn=predict_severity,
    inputs=[
 # Display the image
        gr.File(file_types=['.csv'], label="Upload CSV File")
    ],
    outputs=[
        gr.Image(value="/content/workflow_diagram (1).png", type="filepath", label="Flowchart", width = 206.5, height= 351.5),
        gr.Dataframe(label="Predicted DataFrame"),
        gr.Textbox(label=" Weighted Accuracy"),
        gr.Textbox(label=" Weighted Precision"),
        gr.Textbox(label=" Weighted Recall"),
        gr.Textbox(label=" Weighted F1 Score"),
        gr.File(label="Download Predictions CSV")
    ]
)

# Create the Gradio interface


# Launch the interface
interface.launch()
    
