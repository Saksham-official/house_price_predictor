import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json

app = Flask(__name__)


housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

features_to_use = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
X = X[features_to_use]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
best_model_name = ""
best_model_instance = None
best_r2_score = -1

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        "MSE": f"{mse:.4f}",
        "MAE": f"{mae:.4f}",
        "R2 Score": f"{r2:.4f}"
    }
    
    if r2 > best_r2_score:
        best_r2_score = r2
        best_model_name = name
        best_model_instance = model

print(f"\nBest Model Found: {best_model_name}")



@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    if request.method == 'POST':
        try:
            input_data = [float(request.form[f]) for f in features_to_use]
            
            input_df = pd.DataFrame([input_data], columns=features_to_use)
            
            input_scaled = scaler.transform(input_df)
            
            predicted_price = best_model_instance.predict(input_scaled)[0]
            
            confidence_interval = None
            if isinstance(best_model_instance, RandomForestRegressor):
                individual_tree_predictions = np.array([tree.predict(input_scaled) for tree in best_model_instance.estimators_])
                std_dev = np.std(individual_tree_predictions)
                lower_bound = predicted_price - 1.96 * std_dev
                upper_bound = predicted_price + 1.96 * std_dev
                confidence_interval = {
                    "lower": f"{lower_bound:.2f}",
                    "upper": f"{upper_bound:.2f}"
                }

            prediction_result = {
                "predicted_price": f"{predicted_price:.2f} (in $100,000s)",
                "confidence_interval": confidence_interval
            }
        except Exception as e:
            prediction_result = {"error": str(e)}

    return render_template('index.html', 
                           model_results=results, 
                           best_model=best_model_name, 
                           feature_names=features_to_use, 
                           prediction=prediction_result)

if __name__ == '__main__':
    app.run(debug=True)