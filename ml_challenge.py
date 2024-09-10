import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer
import joblib

# 1. Preprocessing and Loading the Dataset
def load_and_preprocess_data(csv_file):
    # Load the dataset
    df = pd.read_csv(csv_file)

    df = df[['Heading', 'Article.Description', 'Full_Article', 'Article_Type']]

    df['text'] = df['Heading'].fillna('') + ' ' + df['Article.Description'].fillna('') + ' ' + df['Full_Article'].fillna('')

  
    df = df.dropna(subset=['Article_Type'])

    return df

# 2. Vectorization using SentenceBERT
def vectorize_text(df, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    X = model.encode(df['text'].tolist())
    y = df['Article_Type']
    return X, y

# 3. Train and Select a Model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, 30]
    }

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)


    best_rf = grid_search.best_estimator_

  
    y_pred = best_rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(best_rf, 'article_type_model.pkl')

    return best_rf

# 4. Load Model and Predict New Articles
def load_model_and_predict(model_path, new_articles_csv):

    model = joblib.load(model_path)


    df_new = pd.read_csv(new_articles_csv)


    df_new['text'] = df_new['Heading'].fillna('') + ' ' + df_new['Full_Article'].fillna('')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    X_new = model.encode(df_new['text'].tolist())

  
    predictions = model.predict(X_new)

    df_new['Predicted_Article_Type'] = predictions
    df_new.to_csv('predicted_articles.csv', index=False)
    print("Predictions saved to predicted_articles.csv")

# 5. FastAPI for Serving the Model
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.post("/predict_article_type/")
def predict_article_type(heading: str, description: str, full_article: str):

    text = heading + ' ' + description + ' ' + full_article


    model = SentenceTransformer('all-MiniLM-L6-v2')
    vectorized_text = model.encode([text])

  
    classifier = joblib.load('article_type_model.pkl')

    prediction = classifier.predict(vectorized_text)

    return {"Predicted Article Type": prediction[0]}

if _name_ == "_main_":
    uvicorn.run(app, host="0.0.0.0", port=8000)
