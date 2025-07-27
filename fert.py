import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
fertilizer_data = pd.read_csv(r"C:\Users\kowsh\OneDrive\csp project\augmented_crop_conditions_dataset.csv")

# Preprocess data
label_encoders = {
    'Crop': LabelEncoder(),
    'Soil': LabelEncoder(),
    'Fertilizer': LabelEncoder()
}
fertilizer_data['Crop'] = label_encoders['Crop'].fit_transform(fertilizer_data['Crop'])
fertilizer_data['Soil'] = label_encoders['Soil'].fit_transform(fertilizer_data['Soil'])
fertilizer_data['Fertilizer'] = label_encoders['Fertilizer'].fit_transform(fertilizer_data['Fertilizer'])

# Features and target
X = fertilizer_data[['Temperature', 'pH', 'N', 'P', 'K', 'Crop', 'Soil']]
y = fertilizer_data['Fertilizer']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Define parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Perform GridSearchCV with cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

# Train the model with cross-validation
grid_search.fit(X_train, y_train)

# Get the best model
best_rf = grid_search.best_estimator_

# Predict on the test set
y_pred = best_rf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Optimized Model Accuracy: {accuracy * 100:.2f}%")

# Save the optimized model and label encoders
with open('optimized_fertilizer_model.pkl', 'wb') as file:
    pickle.dump(best_rf, file)

for enc_name, encoder in label_encoders.items():
    with open(f"{enc_name}_encoder.pkl", "wb") as file:
        pickle.dump(encoder, file)
