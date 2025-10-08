from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load('random_forest_model.pkl')
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    
        # Extract features from the form
    feature1=float(request.form['feature1'])
    feature2=float(request.form['feature2'])
    feature3=float(request.form['feature3'])
    feature4=float(request.form['feature4'])
    feature5=float(request.form['feature5'])
    feature6=float(request.form['feature6'])
    feature7=float(request.form['feature7'])
    feature8=float(request.form['feature8'])
    feature9=float(request.form['feature9'])
    # create a list of input features
    input_features = [[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9]]
    # Make prediction
    prediction = model.predict(input_features)
    # map the prediction to a human-readable label
    prediction_label = 'Successful' if prediction[0] == 1 else 'Unsuccessful'
    # Render the result template with the prediction
    return render_template('result.html', prediction=prediction_label)
if __name__ == '__main__':
    app.run(debug=True)