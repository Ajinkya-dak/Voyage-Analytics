from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained pipeline
model = joblib.load('flight_price_linreg_pos_pipeline.joblib')

# Valid choices for dropdowns (populate from your data or hardcode)
FROM_OPTIONS       = sorted(model.named_steps['preprocessor']
                          .transformers_[0][1]
                          .categories_[0])
TO_OPTIONS         = sorted(model.named_steps['preprocessor']
                          .transformers_[0][1]
                          .categories_[1])
TYPE_OPTIONS       = sorted(model.named_steps['preprocessor']
                          .transformers_[0][1]
                          .categories_[2])
AGENCY_OPTIONS     = sorted(model.named_steps['preprocessor']
                          .transformers_[0][1]
                          .categories_[3])

@app.route('/', methods=['GET'])
def home():
    return render_template(
        'index.html',
        from_opts=FROM_OPTIONS,
        to_opts=TO_OPTIONS,
        type_opts=TYPE_OPTIONS,
        agency_opts=AGENCY_OPTIONS
    )

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    data = {
        'from':       request.form['from_city'],
        'to':         request.form['to_city'],
        'flightType': request.form['flight_type'],
        'agency':     request.form['agency'],
        'time':       float(request.form['time']),
        'distance':   float(request.form['distance'])
    }
    # DataFrame for model
    df = pd.DataFrame([data])
    # Predict
    pred = model.predict(df)[0]
    # Clamp negative to zero
    pred = max(pred, 0)
    # Format
    price_str = f"₹{pred:,.2f}"
    return render_template(
        'index.html',
        from_opts=FROM_OPTIONS,
        to_opts=TO_OPTIONS,
        type_opts=TYPE_OPTIONS,
        agency_opts=AGENCY_OPTIONS,
        prediction=price_str,
        inputs=data
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
