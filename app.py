from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor

# Open the pickle file for reading
with open('gpa_predict_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    error = None
    if request.method == 'POST':
        try:
            # Get user inputs from the form
            inputs = {
                'Course_LOI': float(request.form.get('Course_LOI', 0)),
                'Rate_of_feeding': float(request.form.get('Rate_of_feeding', 0)),
                'Jamb': float(request.form.get('Jamb', 0)),
                'Study_Rate': float(request.form.get('Study_Rate', 0)),
                'Assimilation_rate': float(request.form.get('Assimilation_rate', 0)),
                'Attendance_Rate': float(request.form.get('Attendance_Rate', 0)),
                'Extra_activities': float(request.form.get('Extra_activities', 0))
            }
            
            # Create a DataFrame with the user inputs
            input_df = pd.DataFrame([inputs])
            
            # Ensure that the input DataFrame has the correct columns
            expected_columns = ['Course_LOI', 'Rate_of_feeding', 'Jamb', 'Study_Rate', 'Assimilation_rate', 'Attendance_Rate', 'Extra_activities']
            for col in expected_columns:
                if col not in input_df.columns:
                    input_df[col] = 0.0  # Fill missing columns with default values if necessary
            
            # Make a prediction using the trained model
            prediction = model.predict(input_df)[0]
        
        except Exception as e:
            # Handle exceptions and return an error message
            error = str(e)
    
    return render_template('index1.html', prediction=prediction, error=error)

if __name__ == '__main__':
    app.run(debug=True)
