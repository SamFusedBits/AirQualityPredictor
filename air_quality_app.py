import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Load and preprocess dataset
@st.cache_resource
def load_and_preprocess_data(filepath):
    data = pd.read_excel(filepath)

    # Handle missing values
    data.replace(-200, np.nan, inplace=True)
    data.drop(columns=['Date', 'Time'], inplace=True)  # Drop Date and Time columns as they are not directly useful
    data.fillna(data.mean(), inplace=True)  # Fill missing values with column means

    return data

# Custom feature scaling
def custom_feature_scaling(X):
    feature_weights = {
        'CO(GT)': 1.5, 'NMHC(GT)': 1.2, 'C6H6(GT)': 1.3,
        'NOx(GT)': 1.5, 'T': 1.1, 'RH': 1.0, 'AH': 1.0
    }

    for feature, weight in feature_weights.items():
        if feature in X.columns:
            X[feature] *= weight

    return X

# Prepare data for training and testing
def prepare_data(data, target_column='NO2(GT)', test_size=0.2):
    selected_features = ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'T', 'RH', 'AH']  # Include only these features
    X = data[selected_features]
    y = data[target_column]

    X = custom_feature_scaling(X)  # Scale features

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    scaler = StandardScaler()
    power_transformer = PowerTransformer()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = power_transformer.fit_transform(X_train_scaled)
    X_test_scaled = power_transformer.transform(X_test_scaled)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, power_transformer

# Build optimized neural network model
def build_model(input_shape):
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(input_shape,), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.4),
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu', kernel_initializer='he_uniform'),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model

# Train the model with callbacks
def train_model(model, X_train, y_train):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
        ModelCheckpoint(filepath='best_air_quality_model.h5', monitor='val_loss', save_best_only=True, mode='min')
    ]

    history = model.fit(
        X_train, y_train, validation_split=0.2, epochs=10, batch_size=64,
        callbacks=callbacks, verbose=2
    )
    return history

# Evaluate model performance
def evaluate_model(model, X_test, y_test):
    test_loss, test_mae = model.evaluate(X_test, y_test)
    return test_loss, test_mae

# Updated real-time prediction function
def real_time_prediction(input_data, scaler, power_transformer):
    model = keras.models.load_model('best_air_quality_model.h5')

    input_scaled = scaler.transform([input_data])  # Scale the input
    input_transformed = power_transformer.transform(input_scaled)  # Transform the input
    prediction = model.predict(input_transformed)

    # Squeeze to get a single float value
    predicted_value = prediction[0][0]

    # Clamp the predicted value to a reasonable range if necessary
    predicted_value = max(predicted_value, 0)  # No negative predictions

    return predicted_value

# Suggested measures based on predicted air quality levels
def suggest_measures(predicted_value):
    if predicted_value < 50:
        return (
            "🌱 **Good Air Quality!**\n"
            "The air quality is satisfactory. You can enjoy outdoor activities with minimal concern. "
            "Consider maintaining this quality by using public transport or cycling when possible."
        )
    elif 50 <= predicted_value < 100:
        return (
            "☁️ **Moderate Air Quality.**\n"
            "Air quality is acceptable, but there may be a risk for some pollutants. "
            "For sensitive individuals, it's advisable to limit prolonged outdoor exertion. "
            "Consider implementing more green spaces in urban areas to help improve air quality."
        )
    elif 100 <= predicted_value < 200:
        return (
            "🚷 **Unhealthy for Sensitive Groups.**\n"
            "People with respiratory or heart conditions should limit outdoor activities. "
            "It's a good time for city planners to enhance regulations on traffic emissions and industrial outputs."
        )
    elif 200 <= predicted_value < 300:
        return (
            "⚠️ **Unhealthy Air Quality.**\n"
            "Health effects may be experienced by everyone, with sensitive groups facing more serious effects. "
            "Consider reducing vehicle usage and promoting carpooling or public transport to alleviate traffic emissions."
        )
    elif 300 <= predicted_value < 400:
        return (
            "😷 **Very Unhealthy.**\n"
            "Health alert! Everyone may experience more serious health effects. "
            "It’s crucial to stay indoors as much as possible. Urban authorities should enforce stricter emissions regulations and consider traffic control measures."
        )
    else:
        return (
            "❗ **Hazardous Air Quality!**\n"
            "Health warnings of emergency conditions. The entire population is more likely to be affected. "
            "Immediate action is required: reduce outdoor activities, and authorities must implement emergency measures to control pollution sources."
        )

# Updated function for bar chart visualization with labels and color coding
def plot_barchart(predicted_value):
    # Define threshold levels and corresponding labels
    thresholds = [50, 100, 200, 300, 400, 500]
    labels = ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous (Values above 400 are dangerous)"]
    colors = ['green', 'yellow', 'orange', 'red', 'purple', 'maroon']

    # Create separate figure for better layout
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the threshold bars
    ax.barh(labels, thresholds, color=colors, edgecolor='black', label="Threshold Levels")

    # Plot the predicted NO2 value separately
    ax.barh(["Predicted NO2 Level"], [predicted_value], color='blue', edgecolor='black', label="Predicted NO2 Level")

    # Set x-axis limit to prevent the predicted NO2 value from going outside the plot
    max_threshold = max(thresholds)
    ax.set_xlim(0, max(predicted_value, max_threshold) * 1.2)  # Extend the x-axis to fit larger values

    # Add labels and title
    ax.set_xlabel("NO2 Level")
    ax.set_title("Predicted NO2 Level vs. Air Quality Thresholds")

    # Add value annotations
    for i, v in enumerate(thresholds):
        ax.text(v + 5, i, str(v), color='black', va='center')
    
    # Adjust the position of the predicted NO2 level text, ensuring it stays within the plot
    # Adjust the predicted NO2 level text position to be closer to the bar
    if predicted_value > 400:
        ax.text(predicted_value * 1.025, len(thresholds), f"{predicted_value:.2f}", 
                color='maroon', va='center', fontweight='bold')
    else:
        ax.text(predicted_value * 0.95, len(thresholds) - 1, f"{predicted_value:.2f}", 
                color='blue', va='center')


    st.pyplot(fig)

# Streamlit UI
st.set_page_config(page_title="Real-Time Air Quality Prediction", layout="wide")
st.title("🌬️ Real-Time Air Quality Prediction")
st.write("This app predicts air quality levels based on user input parameters.")
st.markdown("---")

# User input section with a sidebar for parameters
st.sidebar.header("Input Parameters")
co = st.sidebar.number_input("CO (GT)", min_value=0.0, max_value=500.0, value=0.0)
nmhc = st.sidebar.number_input("NMHC (GT)", min_value=0.0, max_value=500.0, value=0.0)
c6h6 = st.sidebar.number_input("C6H6 (GT)", min_value=0.0, max_value=500.0, value=0.0)
nox = st.sidebar.number_input("NOx (GT)", min_value=0.0, max_value=500.0, value=0.0)
temperature = st.sidebar.number_input("Temperature (T)", min_value=0.0, max_value=500.0, value=0.0)
humidity = st.sidebar.number_input("Relative Humidity (RH)", min_value=0.0, max_value=500.0, value=0.0)
ah = st.sidebar.number_input("Absolute Humidity (AH)", min_value=0.0, max_value=500.0, value=0.0)

# When button is pressed
if st.sidebar.button("Predict"):
    input_data = np.array([co, nmhc, c6h6, nox, temperature, humidity, ah])
    
    # Load the dataset
    data = load_and_preprocess_data("AirQualityUCI.xlsx")
    
    # Prepare the data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, power_transformer = prepare_data(data)
    
    # Build and train the model
    model = build_model(X_train_scaled.shape[1])
    train_model(model, X_train_scaled, y_train)
    
    # Make prediction
    predicted_value = real_time_prediction(input_data, scaler, power_transformer)
    suggestion = suggest_measures(predicted_value)
    
    # Display results in the main area
    st.markdown("---")
    st.subheader("Prediction Results")
    st.write(f"**Predicted NO2 Level:** {predicted_value:.2f}")

    # Visualize the predicted value with a separate bar chart
    st.write("### NO2 Level Comparison")
    plot_barchart(predicted_value)
    
    st.write("### Suggested Measures:")
    st.markdown(suggestion)

# Footer
st.markdown("---")
st.write("Made with ❤️ by KMK")
