from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import os

app = Flask(__name__)

# Load data
data = pd.read_csv("reservation_data.csv")

# Encode categorical variables
data_encoded = pd.get_dummies(data, columns=['source', 'sales_manager', 'interest', 'probability'], drop_first=True)

# Split data into features and target
X = data_encoded.drop(['reservation_success'], axis=1)
y = data_encoded['reservation_success']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a pipeline with polynomial features and logistic regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('logreg', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

# Define hyperparameters to tune
param_grid = {
    'poly__degree': [1, 2, 3],
    'logreg__C': [0.01, 0.1, 1, 10, 100]
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Save the best model
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'best_model.pkl')

# Calculate accuracy on the test set
accuracy_on_test_set = accuracy_score(y_test, best_model.predict(X_test))

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.json

    # Convert JSON to DataFrame
    new_client = pd.DataFrame([data])

    # Encode new client data
    new_client_encoded = pd.get_dummies(new_client, columns=['source', 'sales_manager', 'interest'], drop_first=True)

    # Ensure new client data has the same columns as training data
    training_columns = X.columns
    for column in training_columns:
        if column not in new_client_encoded.columns:
            new_client_encoded[column] = 0  # Add missing column and set to 0

    # Reorder columns to match training data
    new_client_encoded = new_client_encoded[training_columns]

    # Load the best model
    best_model = joblib.load('best_model.pkl')

    # Predict probability for new client
    reservation_probability = best_model.predict_proba(new_client_encoded)
    result = {
        "Seriousness": 'no' if reservation_probability[0][1] < 0.5 else 'yes',
        "probability":  reservation_probability[0][1]*100,
        "accuracy": accuracy_on_test_set*100  # Include accuracy in the response
    }

    file = pd.read_csv("reservation_data.csv")

    interest = data["interest"]
    sales_manager = data["sales_manager"]
    source = data["source"]

    new_client_data = {
        "index": len(file),
        "source": source,
        "sales_manager": sales_manager,
        "interest": interest,
        "probability": result["probability"],
        "reservation_success": result["Seriousness"]
    }

    # Convert new client data to DataFrame
    new_client_df = pd.DataFrame([new_client_data])

    # Append the new customer data to the dataset using pd.concat
    data = pd.concat([file, new_client_df], ignore_index=True)

    # Save the updated dataset to CSV
    data.to_csv("reservation_data.csv", index=False)



    # Return the prediction result as JSON
    return jsonify(result)

@app.route('/update_data', methods=['POST'])
def update_data():
    # Get JSON data from the request
    data = request.json
    file = pd.read_csv("reservation_data.csv")

    interest = data["interest"]
    sales_manager = data["sales_manager"]
    source = data["source"]

    # Predict the reservation probability
    response = predict()
    prediction = response.json

    new_client_data = {
        "index": len(file),
        "source": source,
        "sales_manager": sales_manager,
        "interest": interest,
        "reservation_success": prediction["Seriousness"]
    }

    # Convert new client data to DataFrame
    new_client_df = pd.DataFrame([new_client_data])

    # Append the new customer data to the dataset using pd.concat
    data = pd.concat([file, new_client_df], ignore_index=True)

    # Save the updated dataset to CSV
    data.to_csv("reservation_data.csv", index=False)

    return jsonify({"message": "Data updated successfully"}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
