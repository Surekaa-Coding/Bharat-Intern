import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import joblib

# Load the dataset from Excel workbook
file_path = r'C:\Users\surek\OneDrive\Documents\spamcollectiondataset.xlsx'
df = pd.read_excel(file_path)

# Rename the columns for clarity
df.rename(columns={'v1': 'Label', 'v2': 'Text'}, inplace=True)

# Preprocess the dataset by removing any missing values in 'Text' and 'Label'
df = df.dropna(subset=['Text', 'Label'])

# Convert all entries in the 'Text' column to strings and clean the text
df['Text'] = df['Text'].astype(str).str.lower().str.replace(r'\W', ' ', regex=True)

# Split the data into features (X) and labels (y)
X = df['Text']
y = df['Label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with TfidfVectorizer and MultinomialNB
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('nb', MultinomialNB())
])

# Set up the parameter grid for hyperparameter tuning
param_grid = {
    'tfidf__max_df': [0.5, 0.75, 1.0],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'nb__alpha': [0.1, 0.5, 1.0]
}

# Perform GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the results
print(f"Best Model Parameters: {grid_search.best_params_}\n")
print(f"Model Accuracy: {accuracy:.2f}\n")
print("Classification Report:\n")
print(report)

# Save the best model for future use
joblib.dump(best_model, 'sms_spam_classifier_optimized.pkl')

# Save classification report to a text file
report_file_path = 'classification_report.txt'
with open(report_file_path, 'w') as file:
    file.write(report)

# Save classification report to a CSV file
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
csv_file_path = 'classification_report.csv'
report_df.to_csv(csv_file_path)

# Create a DataFrame for the test set
test_df = pd.DataFrame({'Text': X_test, 'True Label': y_test, 'Predicted Label': y_pred})

# Save the DataFrame with predictions to a new Excel file
output_file_path = r'C:\Users\surek\OneDrive\Documents\spam_predictions.xlsx'
test_df.to_excel(output_file_path, index=False)

print(f"Predictions saved to '{output_file_path}'")
