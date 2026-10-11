# 🚗 Used Car Price Prediction Using Machine Learning

## 1. Project Overview
Used Car Price Prediction is a Machine Learning project that predicts the estimated selling price of a used car based on its details. The project uses Python, Pandas, Scikit-learn, and Streamlit.

## 2. Project Objective
The main objective is to develop a Machine Learning model that estimates used-car prices based on features such as car name, manufacturing year, kilometers driven, fuel type, seller type, transmission, and ownership.

## 3. Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## 4. Dataset
The project uses a used-car dataset containing vehicle details and selling prices. The dataset is cleaned and prepared before training the model.

## 5. Data Preprocessing
- Loaded the dataset using Pandas.
- Removed duplicate rows and missing values.
- Separated input features and the target selling price.
- Converted categorical features using One-Hot Encoding.
- Split the data into training and testing sets.

## 6. Machine Learning Model
A Random Forest Regressor is used to train the model and predict used-car selling prices.

## 7. Model Evaluation
The model was evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score.

- MAE: 150088.47
- MSE: 156077028685.08
- R² Score: 0.5155

These values are from the current model run. The results may vary if the dataset or training settings change.

## 8. Streamlit Application
The Streamlit application allows users to select or enter car details and receive an estimated selling price.

## 9. Project Files
- `car_data.csv` — Dataset
- `train_model.py` — Model training and evaluation
- `app.py` — Streamlit user interface
- `requirements.txt` — Required Python libraries
- `models/` — Saved model and feature columns
- `screenshots/` — Project screenshots
- `actual_vs_predicted.png` — Model evaluation graph

## 10. How to Run the Project
1. Install Python.
2. Open the project folder in VS Code.
3. Install the required libraries:

   `python -m pip install -r requirements.txt`

4. Train the model:

   `python train_model.py`

5. Run the application:

   `python -m streamlit run app.py`

6. Open the local URL displayed in the terminal.

## 11. Conclusion
This project demonstrates how Machine Learning can be used to estimate used-car selling prices. It includes data preprocessing, model training, evaluation, saving the trained model, and developing a Streamlit application for predictions.

The predicted price is an estimate and may differ from the actual market price.