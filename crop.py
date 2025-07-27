import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

crop = pd.read_csv(r"C:\Users\kowsh\OneDrive\csp project\crop recommendation with project.csv")

crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5,
    'papaya': 6, 'orange': 7, 'apple': 8, 'muskmelon': 9, 'watermelon': 10,
    'grapes': 11, 'mango': 12, 'banana': 13, 'pomegranate': 14, 
    'lentil': 15, 'blackgram': 16, 'mungbean': 17, 'mothbeans': 18, 
    'pigeonpeas': 19, 'kidneybeans': 20, 'chickpea': 21, 'coffee': 22,
    'groundnut': 23, 'wheat': 24, 'onion': 25, 'barley': 26, 'clover': 27,
    'oats': 28
}

soil_dict = {
    'Clayey': 1, 'Sandy': 2, 'Loam': 3, 'Silty': 4, 'Peaty': 5,
    'Saline': 6, 'Red': 7, 'Black': 8, 'Alluvial': 9
}

season_dict = {
    'Kharif': 1, 'Rabi': 2
}

crop['crop_num'] = crop['label'].map(crop_dict)
crop.drop(['label'], axis=1, inplace=True)
crop['Soil'] = crop['Soil'].map(soil_dict)
crop['prev_crop'] = crop['prev_crop'].map(crop_dict)
crop['prev_num'] = crop['prev_num'].map({'short': 0, 'long': 1})
crop['season'] = crop['season'].map(season_dict)
crop['rec_num'] = crop['rec_num'].map({'short': 0, 'long': 1})

crop.fillna(crop.mean(), inplace=True)

X = crop.drop(['crop_num'], axis=1)
y = crop['crop_num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=30,        #
    max_depth=8,            
    min_samples_split=10,   
    min_samples_leaf=5,     
    random_state=42         
)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kfold)
print(f"Cross-Validation Accuracy: {np.mean(cv_scores)}")

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy}")

# Save model and scaler
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)


def recommendation(N, P, K, temperature, humidity, ph, rainfall, soil, prev_crop, prev_duration, rec_duration, season):
    soil_num = soil_dict.get(soil, -1)
    prev_crop_num = crop_dict.get(prev_crop.lower(), -1)
    season_num = season_dict.get(season, -1)
    prev_duration_num = 1 if prev_duration.lower() == 'long' else 0
    rec_duration_num = 1 if rec_duration.lower() == 'long' else 0


    if soil_num == -1 or prev_crop_num == -1 or season_num == -1:
        print("Invalid input for soil type, previous crop, or season.")
        return None

    
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall, soil_num, prev_crop_num, prev_duration_num, season_num, rec_duration_num]]).astype(float)
    features = scaler.transform(features)

    
    prediction = model.predict(features)
    recommended_crop = next((name for name, num in crop_dict.items() if num == prediction[0]), None)

    
    if recommended_crop:
        print(f"The recommended crop is: {recommended_crop.capitalize()}")
    else:
        print("Unable to recommend a proper crop.")
    
    return recommended_crop

N = float(input("Enter Nitrogen value: "))
P = float(input("Enter Phosphorus value: "))
K = float(input("Enter Potassium value: "))
temperature = float(input("Enter Temperature: "))
humidity = float(input("Enter Humidity: "))
ph = float(input("Enter pH value: "))
rainfall = float(input("Enter Rainfall: "))
soil = input("Enter soil type (Clayey, Sandy, Loam, etc.): ")
prev_crop = input("Enter previous crop: ")
prev_duration = input("Enter previous crop duration (short/long): ")
rec_duration = input("Enter recommended crop duration (short/long): ")
season = input("Enter season (Kharif/Rabi): ")


recommended_crop = recommendation(N, P, K, temperature, humidity, ph, rainfall, soil, prev_crop, prev_duration, rec_duration, season)
