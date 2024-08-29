import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# Load the data
data = pd.read_excel(r'C:\Users\surek\OneDrive\Documents\Titanic Dataset.xlsx')

# Feature Engineering
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
data['Title'] = data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
data['Title'] = data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 
                                       'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
data['Title'] = data['Title'].replace('Mlle', 'Miss')
data['Title'] = data['Title'].replace('Ms', 'Miss')
data['Title'] = data['Title'].replace('Mme', 'Mrs')

# Drop unnecessary features
data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Handle missing values
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
data['Fare'] = data['Fare'].fillna(data['Fare'].median())

# Encoding categorical features
label = LabelEncoder()
data['Sex'] = label.fit_transform(data['Sex'])
data['Embarked'] = label.fit_transform(data['Embarked'])
data['Title'] = label.fit_transform(data['Title'])

# Splitting the data into training and testing sets
X = data.drop('Survived', axis=1)
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Class weights (for handling imbalanced classes)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(zip(np.unique(y_train), class_weights))

# Pipeline for scaling and model fitting
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GradientBoostingClassifier())  # or use RandomForestClassifier() to try a different model
])

# Hyperparameter tuning using RandomizedSearchCV
param_distributions = {
    'model__n_estimators': [100, 200, 300, 400],
    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'model__max_depth': [3, 4, 5, 6],
    'model__min_samples_split': [2, 5, 10]
}

random_search = RandomizedSearchCV(pipeline, param_distributions, n_iter=20, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

# Best parameters and model evaluation
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Best parameters:", random_search.best_params_)
print("Best cross-validation score:", random_search.best_score_)
print("Final Accuracy:", accuracy_score(y_test, y_pred))
print("Final Classification Report:\n", classification_report(y_test, y_pred))

# Feature Importance (optional)
if hasattr(best_model.named_steps['model'], 'feature_importances_'):
    feature_importances = best_model.named_steps['model'].feature_importances_
    indices = np.argsort(feature_importances)[::-1]
    names = [X.columns[i] for i in indices]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(X.shape[1]), feature_importances[indices])
    plt.xticks(range(X.shape[1]), names, rotation=90)
    plt.show()
