from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a random secret key

# Load models and encoders
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
fertilizer_encoder = pickle.load(open('fertilizer.pkl', 'rb'))
soil_encoder = pickle.load(open('soil_encoder.pkl', 'rb'))
crop_encoder = pickle.load(open('crop_encoder.pkl', 'rb'))

# Dictionaries for encoding
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

# Input validation ranges
input_ranges = {
    'N': (0, 100),
    'P': (0, 100),
    'K': (0, 100),
    'temperature': (0, 50),
    'humidity': (0, 100),
    'ph': (0, 14),
    'rainfall': (0, 300)
}
fertilizer_dict = {
    "Urea": {
        "nutrient_content": "46% Nitrogen (N)",
        "application_rate": "50-100 kg/ha",
        "benefits": "Provides a high concentration of nitrogen, essential for promoting vegetative growth and green foliage.",
        "suitable_crops": "Rice, Wheat, Corn, Sugarcane",
        "usage_tips": "Best applied in the early growth stages, ideally before or right after planting. Avoid overuse to prevent nitrogen leaching."
    },
    "DAP (Diammonium Phosphate)": {
        "nutrient_content": "18% Nitrogen (N), 46% Phosphorus (P2O5)",
        "application_rate": "50-100 kg/ha",
        "benefits": "Provides a good balance of nitrogen and phosphorus, essential for root development and initial crop establishment.",
        "suitable_crops": "Cereals, Oilseeds, Legumes",
        "usage_tips": "Mix well with soil before planting. Avoid direct contact with seeds to prevent damage."
    },
    "MOP (Muriate of Potash)": {
        "nutrient_content": "60% Potassium (K2O)",
        "application_rate": "30-50 kg/ha",
        "benefits": "High potassium content, supports plant water balance and enhances fruit quality.",
        "suitable_crops": "Banana, Potato, Tomato, Sugarcane",
        "usage_tips": "Apply at flowering and fruiting stages to support fruit development. Avoid overuse in soils with high salinity."
    },
    "NPK (Various formulations)": {
        "nutrient_content": "Various formulations (e.g., 15-15-15, 20-20-20)",
        "application_rate": "100-150 kg/ha",
        "benefits": "Balanced nutrient mix for general plant growth and yield improvement.",
        "suitable_crops": "All types of crops, especially vegetables and cereals",
        "usage_tips": "Apply according to soil test results to match crop needs. Ideal for soil amendment in nutrient-poor soils."
    },
    "Calcium Nitrate": {
        "nutrient_content": "15.5% Nitrogen (N), 19% Calcium (Ca)",
        "application_rate": "100-200 kg/ha",
        "benefits": "Improves fruit quality, prevents blossom-end rot in tomatoes and peppers.",
        "suitable_crops": "Tomato, Citrus, Lettuce",
        "usage_tips": "Best applied in split doses throughout the growing season."
    },
    "Magnesium Sulfate (Epsom Salt)": {
        "nutrient_content": "16% Magnesium (Mg), 13% Sulfur (S)",
        "application_rate": "30-50 kg/ha",
        "benefits": "Helps in chlorophyll production and prevents magnesium deficiency, which causes yellowing of leaves.",
        "suitable_crops": "Potato, Tomato, Citrus, Peppers",
        "usage_tips": "Apply as a foliar spray for quick uptake or mix into the soil for gradual release."
    },
    "Ammonium Sulfate": {
        "nutrient_content": "21% Nitrogen (N), 24% Sulfur (S)",
        "application_rate": "50-150 kg/ha",
        "benefits": "Reduces soil pH, providing both nitrogen and sulfur, which are essential for protein synthesis.",
        "suitable_crops": "Rice, Wheat, Cotton, Corn",
        "usage_tips": "Apply during planting, especially in soils with high pH. Avoid excessive use on already acidic soils."
    },
    "Potassium Chloride (KCl)": {
        "nutrient_content": "60% Potassium (K2O)",
        "application_rate": "50-200 kg/ha",
        "benefits": "Supports root development, fruit quality, and disease resistance.",
        "suitable_crops": "Root vegetables, Fruit crops, Legumes",
        "usage_tips": "Ideal for soils with low potassium. Avoid high applications on saline soils."
    },
    "Monoammonium Phosphate (MAP)": {
        "nutrient_content": "11% Nitrogen (N), 52% Phosphorus (P2O5)",
        "application_rate": "50-150 kg/ha",
        "benefits": "High in phosphorus, promotes early root growth and seedling development.",
        "suitable_crops": "Corn, Wheat, Barley",
        "usage_tips": "Incorporate into soil at the time of planting to prevent phosphorus fixation."
    },
    "Single Superphosphate (SSP)": {
        "nutrient_content": "16% Phosphorus (P2O5), 12% Sulfur (S)",
        "application_rate": "100-300 kg/ha",
        "benefits": "Improves soil fertility and increases crop yield. Essential for legumes and pulses.",
        "suitable_crops": "Legumes, Vegetables, Forage crops",
        "usage_tips": "Apply before planting and mix well into the soil."
    },
    "Sulfur": {
        "nutrient_content": "90% Sulfur (S)",
        "application_rate": "30-60 kg/ha",
        "benefits": "Essential for protein synthesis and enzyme activity, improves crop flavor and oil content in oilseeds.",
        "suitable_crops": "Oilseeds, Onion, Garlic, Brassicas",
        "usage_tips": "Can be applied as a soil amendment or foliar spray in crops needing high sulfur."
    },
    "Zinc Sulfate": {
        "nutrient_content": "36% Zinc (Zn)",
        "application_rate": "20-50 kg/ha",
        "benefits": "Essential for enzyme activation and hormone production, prevents stunted growth and leaf chlorosis.",
        "suitable_crops": "Corn, Wheat, Rice, Citrus",
        "usage_tips": "Apply to soil or as a foliar spray when plants show signs of zinc deficiency."
    }
}


def recommendation(N, P, K, temperature, humidity, ph, rainfall, soil, prev_crop, prev_duration, rec_duration, season):
    soil_num = soil_dict.get(soil, -1)
    prev_crop_num = crop_dict.get(prev_crop.lower(), -1)
    season_num = season_dict.get(season, -1)
    prev_duration_num = 1 if prev_duration.lower() == 'long' else 0
    rec_duration_num = 1 if rec_duration.lower() == 'long' else 0

    if soil_num == -1 or prev_crop_num == -1 or season_num == -1:
        return "Invalid input for soil type, previous crop, or season.", None

    features = np.array([[N, P, K, temperature, humidity, ph, rainfall, soil_num, prev_crop_num, prev_duration_num, season_num, rec_duration_num]]).astype(float)
    features = scaler.transform(features)
    prediction = model.predict(features)
    recommended_crop = next((name for name, num in crop_dict.items() if num == prediction[0]), None)

    return None, recommended_crop

def recommend_fertilizer(nitrogen, phosphorus, potassium, soil_type, temperature, crop_type):
    try:
        soil_encoded = soil_encoder.transform([soil_type])[0]
        crop_encoded = crop_encoder.transform([crop_type])[0]

        input_data = pd.DataFrame([[nitrogen, phosphorus, potassium, soil_encoded, temperature, crop_encoded]], 
                                  columns=['N', 'P', 'K', 'Soil', 'Temperature', 'Crop'])

        for i in range(6, 12):  
            input_data[f'Feature{i+1}'] = 0  

        fertilizer_index = model.predict(input_data)[0]
        fertilizer_name = fertilizer_encoder.classes_[fertilizer_index]
        fertilizer_details = fertilizer_dict.get(fertilizer_name, {"message": "No additional information available."})

        return fertilizer_name, fertilizer_details

    except Exception as e:
        print(f"Error in recommendation: {e}")
        return "Error", {"message": "An error occurred in the recommendation system."}


def validate_inputs(N, P, K, temperature, humidity, ph, rainfall, prev_crop, prev_duration):
    for param, (min_val, max_val) in input_ranges.items():
        if not (min_val <= locals()[param] <= max_val):
            return f"{param} should be in the range {min_val} to {max_val}."
    
    short_term_crops = {'rice', 'maize', 'lentil', 'mungbean', 'blackgram', 'jute', 'onion','cotton'}
    if prev_crop.lower() in short_term_crops and prev_duration.lower() == 'long':
        return f"{prev_crop.capitalize()} is a short-term crop, so the previous duration should be 'short'."
    
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/crop_recommendation', methods=['GET', 'POST'])
def crop_recommendation():
    recommended_crop = None
    crop_error = None
    if request.method == 'POST':
        try:
            N = float(request.form['N'])
            P = float(request.form['P'])
            K = float(request.form['K'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])
            soil = request.form['soil']
            prev_crop = request.form['prev_crop']
            prev_duration = request.form['prev_duration']
            rec_duration = request.form['rec_duration']
            season = request.form['season']

            validation_error = validate_inputs(N, P, K, temperature, humidity, ph, rainfall, prev_crop, prev_duration)
            if validation_error:
                crop_error = validation_error
            else:
                crop_error, recommended_crop = recommendation(N, P, K, temperature, humidity, ph, rainfall, soil, prev_crop, prev_duration, rec_duration, season)

        except Exception as e:
            crop_error = str(e)

    return render_template('crop_recommend.html', crop_error=crop_error, recommended_crop=recommended_crop)

@app.route('/fertilizer_recommendation', methods=['GET', 'POST'])
def fertilizer_recommendation():
    fertilizer_name = None
    fertilizer_details = None
    fertilizer_error = None

    if request.method == 'POST':
        try:
            # Get form data and validate required fields
            nitrogen = float(request.form.get('N', 0))
            phosphorus = float(request.form.get('P', 0))
            potassium = float(request.form.get('K', 0))
            soil_type = request.form.get('soil_type', '').strip()  # Ensure field name matches HTML
            temperature = float(request.form.get('Temperature', 0))
            crop_type = request.form.get('Crop', '').strip()

            # Check if required fields are missing
            if not soil_type or not crop_type:
                raise ValueError("Soil Type and Crop Type are required fields.")

            # Call recommendation function
            fertilizer_name, fertilizer_details = recommend_fertilizer(nitrogen, phosphorus, potassium, soil_type, temperature, crop_type)

        except ValueError as ve:
            fertilizer_error = f"Input error: {ve}"
        except Exception as e:
            fertilizer_error = f"An unexpected error occurred: {e}"

    return render_template(
        'fertilizer_recommend.html', 
        fertilizer_name=fertilizer_name, 
        fertilizer_details=fertilizer_details, 
        fertilizer_error=fertilizer_error
    )

fertilizer_info = {
    "Urea": {
        "nutrient_content": "46% N",
        "application_rate": "50-100 kg/ha",
        "benefits": "High in nitrogen, promotes vegetative growth.",
        "suitable_crops": ["Rice", "Wheat", "Corn"],
        "usage_tips": "Best applied in the early growth stages."
    },
    "DAP": {
        "nutrient_content": "18% N, 46% P2O5",
        "application_rate": "50-100 kg/ha",
        "benefits": "Provides phosphorus, enhances root development.",
        "suitable_crops": ["Legumes", "Cereals", "Oilseeds"],
        "usage_tips": "Mix with soil before planting."
    },
    "MOP": {
        "nutrient_content": "60% K2O",
        "application_rate": "30-50 kg/ha",
        "benefits": "Rich in potassium, beneficial for fruits.",
        "suitable_crops": ["Potato", "Tomato", "Banana"],
        "usage_tips": "Apply during flowering and fruiting."
    },
    "NPK": {
        "nutrient_content": "Various formulations (e.g., 15-15-15)",
        "application_rate": "100-150 kg/ha",
        "benefits": "Balanced nutrition for plant growth.",
        "suitable_crops": ["All crops"],
        "usage_tips": "Apply based on soil test recommendations."
    },
    "Calcium Nitrate": {
        "nutrient_content": "15.5% N, 19% Ca",
        "application_rate": "100-200 kg/ha",
        "benefits": "Provides calcium and nitrogen, enhances fruit quality.",
        "suitable_crops": ["Fruits", "Vegetables"],
        "usage_tips": "Apply in split doses."
    },
    "Magnesium Sulfate": {
        "nutrient_content": "16% Mg, 13% S",
        "application_rate": "30-50 kg/ha",
        "benefits": "Helps in chlorophyll production.",
        "suitable_crops": ["Potato", "Tomatoes", "Citrus"],
        "usage_tips": "Dissolve in water for foliar spray."
    },
    "Ammonium Sulfate": {
        "nutrient_content": "21% N, 24% S",
        "application_rate": "50-150 kg/ha",
        "benefits": "Good source of nitrogen and sulfur, improves soil acidity.",
        "suitable_crops": ["Corn", "Cotton", "Wheat"],
        "usage_tips": "Apply during planting."
    },
    "Potassium Chloride": {
        "nutrient_content": "60% K2O",
        "application_rate": "50-200 kg/ha",
        "benefits": "High potassium content, promotes fruit and root quality.",
        "suitable_crops": ["Potatoes", "Tomatoes", "Apples"],
        "usage_tips": "Can be applied before planting or as a side-dressing."
    },
    "Monoammonium Phosphate (MAP)": {
        "nutrient_content": "11% N, 52% P2O5",
        "application_rate": "50-150 kg/ha",
        "benefits": "High in phosphorus, enhances early root growth.",
        "suitable_crops": ["Corn", "Wheat", "Barley"],
        "usage_tips": "Incorporate into the soil before planting."
    },
    "Superphosphate": {
        "nutrient_content": "20% P2O5",
        "application_rate": "100-300 kg/ha",
        "benefits": "Improves soil fertility and enhances crop yield.",
        "suitable_crops": ["All crops"],
        "usage_tips": "Mix well with soil to prevent fixation."
    },
    "Sulfur": {
        "nutrient_content": "90% S",
        "application_rate": "30-60 kg/ha",
        "benefits": "Essential for protein synthesis and enzyme function.",
        "suitable_crops": ["Oilseeds", "Vegetables", "Cereals"],
        "usage_tips": "Can be applied as a foliar spray or in the soil."
    },
    "Zinc Sulfate": {
        "nutrient_content": "36% Zn",
        "application_rate": "20-50 kg/ha",
        "benefits": "Essential for chlorophyll formation, prevents leaf chlorosis.",
        "suitable_crops": ["Corn", "Wheat", "Rice"],
        "usage_tips": "Best applied with soil moisture."
    },
}


def recommend_fertilizer(nitrogen, phosphorus, potassium, soil_type, temperature, crop_type):

    fertilizer_name = None
    fertilizer_details = {}

    # Check for high nutrient levels to recommend specific fertilizers
    if nitrogen >= 20:
        fertilizer_name = "Urea"  # High nitrogen content
        fertilizer_details = {
            "nutrient_content": "46% Nitrogen (N)",
            "application_rate": "50-100 kg/ha",
            "benefits": "High in nitrogen, promotes vegetative growth and foliage development.",
            "usage_tips": "Apply at the early growth stages for best results.",
            "suitable_crops": ["rice", "wheat", "maize", "cereals", "sugarcane", "millet"]
        }
    elif phosphorus >= 25:
        fertilizer_name = "DAP (Diammonium Phosphate)"  # High phosphorus content
        fertilizer_details = {
            "nutrient_content": "18% Nitrogen (N), 46% Phosphorus (P2O5)",
            "application_rate": "50-100 kg/ha",
            "benefits": "Promotes root development and enhances crop establishment.",
            "usage_tips": "Incorporate into the soil at planting to prevent seed damage.",
            "suitable_crops": ["legumes", "oilseeds", "pulses", "sunflower", "soybean"]
        }
    elif potassium >= 25:
        fertilizer_name = "MOP (Muriate of Potash)"  # High potassium content
        fertilizer_details = {
            "nutrient_content": "60% Potassium (K2O)",
            "application_rate": "30-50 kg/ha",
            "benefits": "Improves fruit quality and disease resistance.",
            "usage_tips": "Best applied during flowering and fruiting stages.",
            "suitable_crops": ["fruits", "vegetables", "sugarcane", "banana", "potato", "grapes"]
        }
    else:
        # Suggest a balanced fertilizer if no specific high nutrient requirement
        fertilizer_name = "NPK (Balanced formulation)"
        fertilizer_details = {
            "nutrient_content": "Balanced formulation, e.g., 15-15-15",
            "application_rate": "100-150 kg/ha",
            "benefits": "Provides a balanced nutrient mix for general growth.",
            "usage_tips": "Apply based on soil test recommendations for best results.",
            "suitable_crops": ["most vegetables", "fruits", "grain crops", "corn", "onions", "barley"]
        }

    # Soil type-specific recommendations
    if soil_type.lower() == "acidic":
        fertilizer_name = "Ammonium Sulfate"
        fertilizer_details = {
            "nutrient_content": "21% Nitrogen (N), 24% Sulfur (S)",
            "application_rate": "50-150 kg/ha",
            "benefits": "Provides sulfur and nitrogen while reducing soil pH.",
            "usage_tips": "Best for acidic soils; apply at planting.",
            "suitable_crops": ["tea", "coffee", "blueberries", "potatoes", "spinach", "peppers"]
        }
    elif soil_type.lower() == "Alluvial":
        fertilizer_name = "Gypsum"
        fertilizer_details = {
            "nutrient_content": "22% Calcium (Ca), 18% Sulfur (S)",
            "application_rate": "100-200 kg/ha",
            "benefits": "Reduces soil alkalinity and improves soil structure.",
            "usage_tips": "Apply before planting; helps with soil permeability.",
            "suitable_crops": ["cotton", "peanuts", "corn", "alfalfa", "soybeans", "wheat"]
        }
    elif soil_type.lower() == "sandy":
        fertilizer_name = "Organic Matter (Compost)"
        fertilizer_details = {
            "nutrient_content": "Varies, rich in organic matter",
            "application_rate": "1-2 tons/ha",
            "benefits": "Improves water retention and nutrient availability.",
            "usage_tips": "Incorporate well into the soil before planting.",
            "suitable_crops": ["watermelon", "pumpkin", "carrots", "lettuce", "tomatoes", "cucumbers"]
        }
    elif soil_type.lower() == "clayey":
        fertilizer_name = "Calcium Nitrate"
        fertilizer_details = {
            "nutrient_content": "15% Nitrogen (N), 19% Calcium (Ca)",
            "application_rate": "50-100 kg/ha",
            "benefits": "Improves soil structure and nutrient uptake in clay soils.",
            "usage_tips": "Apply during active growth stages.",
            "suitable_crops": ["beans", "cabbage", "peppers", "broccoli", "spinach", "lettuce"]
        }

    if temperature < 15:
        fertilizer_name = "Ammonium Nitrate"
        fertilizer_details = {
            "nutrient_content": "34% Nitrogen (N)",
            "application_rate": "50-100 kg/ha",
            "benefits": "Quick-release nitrogen, effective in colder temperatures.",
            "usage_tips": "Apply in early spring when temperatures are low.",
            "suitable_crops": ["wheat", "barley", "rye"]
        }
    elif temperature > 30:
        fertilizer_name = "Slow-release NPK"
        fertilizer_details = {
            "nutrient_content": "20-20-20 (balanced slow-release)",
            "application_rate": "50-150 kg/ha",
            "benefits": "Slow nutrient release prevents burn in hot climates.",
            "usage_tips": "Apply to minimize nitrogen loss due to heat.",
            "suitable_crops": ["corn", "soybeans", "cotton"]
        }


    return fertilizer_name, fertilizer_details


if __name__ == '__main__':
    app.run(debug=True)
