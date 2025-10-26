from flask import Flask, request, render_template
import joblib
app = Flask(__name__)
model = joblib.load('random_forest_model1.pkl')
@app.route('/')
def home():
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
    """canada = float(request.form.get('feature9', 0))
    newyork = float(request.form.get('feature10', 0))
    ma= float(request.form.get('feature11', 0))
    texas = float(request.form.get('feature12', 0))
    other_states = float(request.form.get('feature13', 0))
    vc= float(request.form.get('feature14', 0))
    angel = float(request.form.get('feature15', 0))
    roundA = float(request.form.get('feature16', 0))
    roundB = float(request.form.get('feature17', 0))
    roundC = float(request.form.get('feature18', 0))
    roundD = float(request.form.get('feature19', 0))"""
    avg_participants = float(request.form.get('feature20', 0))
    # top500= float(request.form.get('feature21', 0))
    # create a list of input features
    #input_features = [age_first_funding_year, age_last_funding_year, age_first_milestone_year, age_last_milestone_year, relationship, funding_rounds, funding_total_usd, milestones,canada,newyork,ma,texas,other_states,vc,angel,roundA,roundB,roundC,roundD,avg_participants,top500]
    input_features = [age_first_funding_year, age_last_funding_year, age_first_milestone_year, age_last_milestone_year, relationship, funding_rounds, funding_total_usd, milestones,avg_participants]
    # Make prediction
    prediction = model.predict([input_features])[0]
    # map the prediction to a human-readable label
    if prediction == 1:
        result='Successful'
    else:
        result='Unsuccessful'
    # Render the result template with the prediction
    return render_template('result.html', result=result)
if __name__ == '__main__':
    app.run(debug=True)