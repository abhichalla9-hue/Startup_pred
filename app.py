from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load('model_startup.pkl')
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    
        # Extract features from the form
    age_first_funding_year = float(request.form.get('feature1', 0))
    age_last_funding_year = float(request.form.get('feature2', 0))
    age_first_milestone_year = float(request.form.get('feature3', 0))
    age_last_milestone_year = float(request.form.get('feature4', 0))
    relationship = float(request.form.get('feature5', 0))
    funding_rounds = float(request.form.get('feature6', 0))
    funding_total_usd = float(request.form.get('feature7', 0))
    milestones = float(request.form.get('feature8', 0))
    avg_participants = float(request.form.get('feature9', 0))
    # create a list of input features
    input_features = [[age_first_funding_year, age_last_funding_year, age_first_milestone_year, age_last_milestone_year, relationship, funding_rounds, funding_total_usd, milestones, avg_participants]]
    # Make prediction
    prediction = model.predict(input_features)
    # map the prediction to a human-readable label
    prediction_label = 'Acquired' if prediction[0] == 1 else 'closed'
    # Render the result template with the prediction
    return render_template('result.html', prediction=prediction_label)
if __name__ == '__main__':
    app.run(debug=True)