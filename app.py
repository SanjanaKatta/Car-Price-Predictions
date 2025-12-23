from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# ================= LOAD MODELS =================
with open("brand_price_map.pkl", "rb") as f:
    brand_map = pickle.load(f)

with open("global_mean.pkl", "rb") as f:
    global_mean = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("linear_regression_model.pkl", "rb") as f:
    model = pickle.load(f)


# ================= HELPERS =================
def preprocess_input(form_data):
    """
    Converts user input → model-ready dataframe
    """
    # Extract brand mean
    brand = form_data["Car_Name"].split()[0].lower()
    brand_mean = brand_map.get(brand, global_mean)

    # Map categorical fields
    fuel_map = {"Petrol": 1, "Diesel": 0, "CNG": 2}
    seller_map = {"Dealer": 1, "Individual": 0}
    transmission_map = {"Manual": 1, "Automatic": 0}

    # Build dataframe
    data = {
        "Year": [int(form_data["Year"])],
        "Present_Price": [float(form_data["Present_Price"])],
        "Kms_Driven": [int(form_data["Kms_Driven"])],
        "Fuel_Type": [fuel_map[form_data["Fuel_Type"]]],
        "Seller_Type": [seller_map[form_data["Seller_Type"]]],
        "Transmission": [transmission_map[form_data["Transmission"]]],
        "Owner": [int(form_data["Owner"])],
        "Brand_mean": [brand_mean]
    }

    df = pd.DataFrame(data)

    # Scale features
    df_scaled = scaler.transform(df)

    return df_scaled


# ================= ROUTES =================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        X = preprocess_input(request.form)
        prediction = round(model.predict(X)[0], 2)
        if prediction < 0:
            prediction = 0  # Avoid negative selling price

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
