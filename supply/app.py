from flask import Flask, request, render_template
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load the Random Forest model
model_path = 'A:/supply/best_rf_model.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
else:
    model = None

# Load the dataset
dataset_path = 'A:/supply/electronics_supply_chain_data_500.csv'
if os.path.exists(dataset_path):
    df = pd.read_csv(dataset_path)
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['availability_date'] = pd.to_datetime(df['availability_date'])
else:
    df = None

@app.route('/', methods=['GET'])
def home():
    """
    Render the prediction form (HTML page).
    """
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict product availability based on user input and return the result on the HTML page.
    """
    if model is None or df is None:
        return render_template('predict.html', error='Model or dataset not available')

    try:
        # Retrieve product name and product category from the request form
        product_name = request.form.get('product_name')
        product_category = request.form.get('product_category')  # Get product category from the form
        
        # Fetch product data from the dataset based on the product name and category
        product_data = df[(df['product_name'] == product_name) & (df['product_category'] == product_category)]
        if product_data.empty:
            return render_template('predict.html', error='Product not found in dataset')

        # Extract the first matching row of the product
        product_row = product_data.iloc[0]

        # Prepare input features
        features = {
            'product_category': product_row['product_category'],
            'supplier_reliability': product_row['supplier_reliability'],
            'demand_forecast': product_row['demand_forecast'],
        }
        
        # Convert features to DataFrame and one-hot encode any categorical variables
        features_df = pd.DataFrame([features])
        features_df = pd.get_dummies(features_df)

        # Align features with the model
        model_columns = model.feature_names_in_  # Features the model expects
        missing_cols = set(model_columns) - set(features_df.columns)
        for col in missing_cols:
            features_df[col] = 0  # Add missing columns with zeros

        # Reorder columns to match the model's input
        features_df = features_df[model_columns]

        # Make the prediction
        predicted_availability_days = model.predict(features_df)[0]
        quantity_available = product_row['quantity_available']

        # Convert predicted days to availability date
        predicted_availability_date = product_row['order_date'] + pd.to_timedelta(predicted_availability_days, unit='days')

        # Get supplier details
        supplier_name = product_row['supplier_name']
        supplier_location = product_row['supplier_location']
        demand_forecast = product_row['demand_forecast']

        # Render the result back in the HTML template
        return render_template('predict.html', result={
            'predicted_availability_date': predicted_availability_date.date(),
            'quantity_available': quantity_available,
            'supplier_name': supplier_name,
            'supplier_location': supplier_location,
            'demand_forecast': demand_forecast  # Include demand forecast
        })

    except Exception as e:
        # Handle errors and display them on the HTML page
        return render_template('predict.html', error=f"An error occurred: {e}")

if __name__ == '__main__':
    app.run(debug=True)
