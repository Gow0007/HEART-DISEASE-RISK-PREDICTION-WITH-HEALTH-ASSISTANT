from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os
import json
from datetime import datetime

app = Flask(__name__)

# Load the model
model_path = 'heart_disease_model.pkl'

# Check if model exists, if not train and save it
if not os.path.exists(model_path):
    print("Training new model...")
    # Load the dataset
    df = pd.read_csv('heart.csv')
    
    # Split data into features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model
    pickle.dump(model, open(model_path, 'wb'))
    print("Model training complete!")
else:
    # Load the existing model
    model = pickle.load(open(model_path, 'rb'))

# Healthy ranges for features
healthy_ranges = {
    'age': (20, 80),
    'trestbps': (90, 120),
    'chol': (125, 200),
    'thalach': (60, 100),
    'oldpeak': (0, 1)
}

# Chatbot knowledge base
chatbot_knowledge = {
    "hello": "Hello! I'm your heart health assistant. How can I help you today?",
    "hi": "Hi there! What would you like to know about heart health?",
    "heart disease": "Heart disease refers to various conditions affecting the heart. Risk factors include high blood pressure, high cholesterol, smoking, and diabetes.",
    "risk factors": "Common risk factors for heart disease include: high blood pressure, high cholesterol, smoking, diabetes, obesity, poor diet, and physical inactivity.",
    "prevention": "To prevent heart disease: 1) Eat healthy 2) Exercise regularly 3) Maintain healthy weight 4) Don't smoke 5) Limit alcohol 6) Manage stress",
    "symptoms": """Common heart disease symptoms include:
- Chest pain or discomfort (angina)
- Shortness of breath
- Pain in arms, neck, jaw, shoulder or back
- Nausea/vomiting
- Fatigue
- Dizziness
- Irregular heartbeat
- Swelling in legs/ankles""",
    "left arm pain": """Left arm pain can sometimes indicate heart issues, especially if accompanied by:
- Chest discomfort
- Shortness of breath
- Nausea
- Dizziness

Possible causes:
1. Heart-related: Angina or heart attack
2. Musculoskeletal: Strain or injury
3. Nerve-related: Pinched nerve

If sudden/severe or with other symptoms, seek medical attention immediately.""",
    "left hand pain": """Left hand pain can have various causes:
1. Heart-related (if accompanied by chest pain, nausea, etc.)
2. Carpal tunnel syndrome
3. Arthritis
4. Injury or overuse

Note: Left arm/hand pain can sometimes signal heart problems, especially in women. If persistent or with other symptoms, consult a doctor.""",
    "blood pressure": "Normal blood pressure is below 120/80 mmHg. High blood pressure (hypertension) is 130/80 or higher.",
    "cholesterol": "Healthy total cholesterol is below 200 mg/dL. LDL should be below 100 mg/dL, HDL above 60 mg/dL is good.",
    "diet": "Heart-healthy diet tips: 1) More fruits/vegetables 2) Whole grains 3) Lean proteins 4) Healthy fats 5) Less salt/sugar 6) Limit processed foods",
    "exercise": "For heart health: Aim for 150 mins moderate exercise weekly (like brisk walking) plus muscle-strengthening 2 days/week.",
    "thanks": "You're welcome! Let me know if you have any other questions about heart health.",
    "thank you": "You're welcome! Stay heart healthy!",
    "default": "I'm not sure I understand. Could you ask about heart disease risk factors, prevention, symptoms, or treatments?"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = request.form.to_dict()
        
        # Convert to numpy array in correct order
        features = np.array([
            int(data['age']),
            int(data['sex']),
            int(data['cp']),
            int(data['trestbps']),
            int(data['chol']),
            int(data['fbs']),
            int(data['restecg']),
            int(data['thalach']),
            int(data['exang']),
            float(data['oldpeak']),
            int(data['slope']),
            int(data['ca']),
            int(data['thal'])
        ]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        probability = model.predict_proba(features)[0][1]
        
        # Generate recommendations based on probability
        recommendations = generate_recommendations(probability, features[0])
        
        # Prepare result
        result = {
            'prediction': int(prediction[0]),
            'probability': float(probability),
            'message': 'High risk of heart disease detected' if prediction[0] == 1 else 'Low risk of heart disease',
            'recommendations': recommendations,
            'feature_analysis': generate_feature_analysis(features[0])
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '').lower()
        
        # Enhanced response logic
        response = None
        
        # Check for pain-related keywords
        pain_keywords = ['pain', 'ache', 'hurt', 'discomfort']
        arm_keywords = ['arm', 'hand', 'shoulder', 'left']
        
        if any(word in user_message for word in pain_keywords) and any(word in user_message for word in arm_keywords):
            if 'left' in user_message:
                response = chatbot_knowledge['left arm pain']
            else:
                response = "Arm pain can have various causes. If it's left arm pain, it could sometimes relate to heart health."
        
        # Check for other specific keywords
        for keyword in chatbot_knowledge:
            if keyword in user_message and keyword not in ['default', 'left arm pain']:
                response = chatbot_knowledge[keyword]
                break
        
        # Fallback to default if no specific response found
        if not response:
            response = chatbot_knowledge['default']
        
        # Add timestamp to response
        timestamp = datetime.now().strftime("%H:%M")
        response = f"{response}\n\n[Last updated: {timestamp}]"
        
        return jsonify({
            'response': response,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

def generate_recommendations(probability, features):
    age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal = features
    
    recommendations = {
        'general': [],
        'diet': [],
        'exercise': [],
        'lifestyle': [],
        'warning': None,
        'specific_factors': []
    }
    
    if probability >= 0.7:
        risk_level = "high"
        recommendations['warning'] = "⚠️ You have a high risk of heart disease. Please consult a doctor immediately."
    elif probability >= 0.4:
        risk_level = "moderate"
        recommendations['warning'] = "⚠️ You have a moderate risk of heart disease. Consider consulting a doctor."
    else:
        risk_level = "low"
    
    # General recommendations
    recommendations['general'].append(f"Based on your {risk_level} risk level, regular health checkups are recommended.")
    recommendations['general'].append("Monitor your blood pressure and cholesterol levels regularly.")
    
    # Diet recommendations
    recommendations['diet'].append("Increase intake of fruits, vegetables, and whole grains")
    recommendations['diet'].append("Choose lean proteins like fish, poultry, beans, and legumes")
    recommendations['diet'].append("Limit saturated fats (red meat, full-fat dairy) and trans fats (fried foods, baked goods)")
    
    # Exercise recommendations
    recommendations['exercise'].append("Aim for at least 150 minutes of moderate exercise per week (brisk walking, swimming)")
    recommendations['exercise'].append("Include strength training 2 days per week")
    
    # Lifestyle recommendations
    recommendations['lifestyle'].append("Avoid smoking and limit alcohol consumption")
    recommendations['lifestyle'].append("Maintain a healthy weight (BMI between 18.5-24.9)")
    recommendations['lifestyle'].append("Manage stress through meditation, deep breathing, or hobbies")
    
    # Specific factors
    if trestbps > 130:
        recommendations['specific_factors'].append(f"Your blood pressure ({trestbps} mmHg) is above normal range (90-120 mmHg).")
    if chol > 200:
        recommendations['specific_factors'].append(f"Your cholesterol level ({chol} mg/dL) is above recommended level (<200 mg/dL).")
    if fbs == 1:
        recommendations['specific_factors'].append("Your elevated fasting blood sugar suggests prediabetes/diabetes risk.")
    if exang == 1:
        recommendations['specific_factors'].append("Exercise-induced angina is a significant warning sign for heart disease.")
    if oldpeak > 2:
        recommendations['specific_factors'].append(f"Your ST depression value ({oldpeak}) indicates possible heart stress.")
    
    return recommendations

def generate_feature_analysis(features):
    analysis = []
    age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal = features
    
    # Blood pressure analysis
    if trestbps < 90:
        analysis.append("Your blood pressure is lower than normal (hypotension).")
    elif trestbps <= 120:
        analysis.append("Your blood pressure is in the normal range.")
    elif trestbps <= 129:
        analysis.append("Your blood pressure is elevated.")
    elif trestbps <= 139:
        analysis.append("Your blood pressure indicates stage 1 hypertension.")
    else:
        analysis.append("Your blood pressure indicates stage 2 hypertension.")
    
    # Cholesterol analysis
    if chol < 125:
        analysis.append("Your cholesterol level is very low (consult your doctor).")
    elif chol <= 200:
        analysis.append("Your cholesterol level is in the desirable range.")
    elif chol <= 239:
        analysis.append("Your cholesterol level is borderline high.")
    else:
        analysis.append("Your cholesterol level is high.")
    
    # Heart rate analysis
    if thalach < 60:
        analysis.append("Your maximum heart rate is below normal (bradycardia).")
    elif thalach <= 100:
        analysis.append("Your maximum heart rate is in the normal range.")
    else:
        analysis.append("Your maximum heart rate is above normal (tachycardia).")
    
    return analysis

@app.route('/validate', methods=['POST'])
def validate_input():
    data = request.json
    field = data.get('field')
    value = data.get('value')
    
    validations = {
        'age': lambda v: 20 <= int(v) <= 100 or "Age should be between 20-100",
        'trestbps': lambda v: 80 <= int(v) <= 200 or "BP should be between 80-200 mmHg",
        'chol': lambda v: 100 <= int(v) <= 600 or "Cholesterol should be between 100-600 mg/dL",
        'thalach': lambda v: 70 <= int(v) <= 220 or "Heart rate should be between 70-220 bpm",
        'oldpeak': lambda v: 0 <= float(v) <= 10 or "ST depression should be between 0-10"
    }
    
    if field in validations:
        validation = validations[field](value)
        if isinstance(validation, str):
            return jsonify({'valid': False, 'message': validation})
    
    return jsonify({'valid': True})

if __name__ == '__main__':
    app.run(debug=True)