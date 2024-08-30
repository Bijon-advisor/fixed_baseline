import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from scipy.fft import fft, fftfreq
import MetaTrader5 as mt5
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

# Constants
LOOKBACK = 288
SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
FEATURE_COLUMNS = ['open', 'high', 'low', 'close', 'tick_volume_log', 'high_low', 'close_open']
PREDICTION_INTERVAL = 60  # Time in seconds between predictions
THRESHOLD_PERCENTAGE = 0.004  # Threshold percentage for upper and lower lines


# Unified custom metrics function
def custom_metrics(y_true, y_pred):
    y_true = K.cast(y_true, dtype='int32')
    y_pred = K.cast(y_pred, dtype='float32')

    # Sparse categorical cross-entropy loss
    base_loss = K.sparse_categorical_crossentropy(y_true, y_pred)

    y_pred_int = K.cast(K.argmax(y_pred, axis=-1), dtype='int32')

    # Calculate the actual rise, fall, and consolidate counts
    actual_rise = K.cast(K.sum(K.cast(y_true == 2, 'int32')), 'float32')
    actual_fall = K.cast(K.sum(K.cast(y_true == 0, 'int32')), 'float32')
    actual_total = K.cast(K.shape(y_true)[0], 'float32')

    # Calculate the predicted rise, fall, and consolidate counts
    correct_predicted_rise = K.cast(K.sum(K.cast(y_pred_int == 2, 'int32') & K.cast(y_true == 2, 'int32')), 'float32')
    correct_predicted_fall = K.cast(K.sum(K.cast(y_pred_int == 0, 'int32') & K.cast(y_true == 0, 'int32')), 'float32')

    # Calculate the predicted percentages of rise and fall
    pred_rise_pct = correct_predicted_rise / (actual_rise + K.epsilon())  # Adding epsilon to avoid division by zero
    pred_fall_pct = correct_predicted_fall / (actual_fall + K.epsilon())

    rise_fall_pct = (correct_predicted_rise + correct_predicted_fall) / (actual_rise + actual_fall + K.epsilon())

    # Penalty is based on how close the predicted rise and fall percentages are to the actual rise and fall
    penalty = rise_fall_pct

    # Custom loss
    custom_loss = K.mean(base_loss) + (1 - penalty)  # Scaling the penalty to balance with base_loss

    # Custom accuracy
    custom_accuracy = rise_fall_pct

    return custom_loss, custom_accuracy


# Custom loss function
def custom_loss(y_true, y_pred):
    return custom_metrics(y_true, y_pred)[0]


# Custom accuracy function
def custom_accuracy(y_true, y_pred):
    return custom_metrics(y_true, y_pred)[1]


# Initialize MetaTrader 5
def initialize_mt5():
    if not mt5.initialize(login=51226521, server="Pepperstone-MT5-Live01", password="9Tda9309!"):
        print("initialize() failed, error code =", mt5.last_error())
        quit()


# Fetch the most recent bars
def fetch_recent_bars(symbol, timeframe, num_bars):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    if rates is None:
        print("No data fetched.")
        return None
    df = pd.DataFrame(rates)
    print(f"Fetched {len(df)} bars")
    return df


# Apply rolling Fourier transform
def apply_fourier_transform(segment):
    fft_values = fft(segment)
    magnitudes = np.abs(fft_values)
    frequencies = fftfreq(len(segment), d=1 / (1 / 5))
    return magnitudes, frequencies


# Add features
def add_features(df):
    df['high_low'] = df['high'] - df['low']
    df['close_open'] = df['close'] - df['open']
    df['tick_volume_log'] = np.log(df['tick_volume'] + 1)
    return df


# Scale features
def scale_features(df, feature_columns):
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[feature_columns])
    return scaled_features, scaler


# Prepare data for the transformer model
def prepare_data_for_prediction(df):
    latest_segment = df[-LOOKBACK:].copy()
    latest_segment = add_features(latest_segment)

    # Scale features
    scaled_segment, scaler = scale_features(latest_segment, FEATURE_COLUMNS)
    latest_segment[FEATURE_COLUMNS] = scaled_segment

    # Apply Fourier Transform to the window
    fft_magnitudes, fft_frequencies = apply_fourier_transform(latest_segment['close'].values)

    # Combine the original features with Fourier Transform features
    combined_features = np.hstack(
        [latest_segment.values, fft_magnitudes.reshape(-1, 1), fft_frequencies.reshape(-1, 1)])
    return combined_features


# Make direction predictions
def direction_predictions(X_scaled):
    model = load_model('G:/fourier/execute/288_72.h5',
                       custom_objects={'custom_loss': custom_loss, 'custom_accuracy': custom_accuracy})
    if len(X_scaled) < LOOKBACK:
        return None
    X_scaled = np.array([X_scaled])
    drt_prediction = model.predict(X_scaled)
    return drt_prediction


# Plot the probabilities variation over time
def plot_probabilities(fig, ax, predictions):
    ax.clear()
    predictions = np.array(predictions[-60:])  # Show only the last 60 predictions
    times = range(len(predictions))

    ax.plot(times, predictions[:, 0], label='Fall', color='red')
    ax.plot(times, predictions[:, 1], label='Consolidate', color='blue')
    ax.plot(times, predictions[:, 2], label='Rise', color='green')
    ax.set_xlabel('Time (iterations)')
    ax.set_ylabel('Probability')
    ax.set_title('Probabilities Variation Over Time')
    ax.legend()
    ax.grid(True)


# Plot the OHLC data with thresholds
def plot_ohlc(fig, ax, df):
    ax.clear()
    ax.plot(df.index, df['open'], label='Open', color='blue')
    ax.plot(df.index, df['high'], label='High', color='green')
    ax.plot(df.index, df['low'], label='Low', color='red')
    ax.plot(df.index, df['close'], label='Close', color='black')

    current_close = df['close'].iloc[-1]
    upper_threshold = current_close * (1 + THRESHOLD_PERCENTAGE)
    lower_threshold = current_close * (1 - THRESHOLD_PERCENTAGE)

    ax.axhline(upper_threshold, color='magenta', linestyle='--', label='Upper Threshold')
    ax.axhline(lower_threshold, color='orange', linestyle='--', label='Lower Threshold')

    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.set_title('OHLC Data with Thresholds')
    ax.legend()
    ax.grid(True)


# Main function to fetch real-time data, make predictions, and update the plot
def main():
    print("Initializing MetaTrader 5...")
    initialize_mt5()

    predictions = []

    # Set up the Tkinter window
    root = tk.Tk()
    root.title("Real-Time Prediction Probabilities and OHLC Data")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

    def update_plot():
        print("Fetching recent bars...")
        df = fetch_recent_bars(SYMBOL, TIMEFRAME, 300)
        if df is None or df.empty:
            print("No data or insufficient data fetched.")
            root.after(PREDICTION_INTERVAL * 1000, update_plot)
            return

        print("Data fetched successfully.")
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.set_index('time')

        print("Preparing data for prediction...")
        X = prepare_data_for_prediction(df)

        print("Making direction predictions...")
        drct_predictions = direction_predictions(X)

        if drct_predictions is not None:
            print(f"Direction prediction: {drct_predictions}")
            predictions.append(drct_predictions[0])  # Append the probabilities
            plot_probabilities(fig, ax1, predictions)
            plot_ohlc(fig, ax2, df)
            canvas.draw()
        else:
            print("Not enough data for prediction or prediction failed.")

        root.after(PREDICTION_INTERVAL * 1000, update_plot)

    # Start the first iteration
    update_plot()

    root.mainloop()


# Fetch real-time data, make predictions, and update the plot
if __name__ == "__main__":
    main()
