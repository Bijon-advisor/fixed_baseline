import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Dense, Dropout, Input, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from scipy.fft import fft, fftfreq
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tqdm import tqdm
import joblib
import tkinter as tk
from tkinter import scrolledtext
from tensorflow.keras.callbacks import Callback

# Initialize MetaTrader 5
def initialize_mt5():
    if not mt5.initialize(login=51226521, server="Pepperstone-MT5-Live01", password="9Tda9309!"):
        print("initialize() failed, error code =", mt5.last_error())
        quit()

# Fetch historical data
def fetch_data(symbol, timeframe, start_date, end_date):
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None:
        print("No data fetched.")
        return None
    return pd.DataFrame(rates)

def add_features(df):
    df['high_low'] = df['high'] - df['low']
    df['close_open'] = df['close'] - df['open']
    df['tick_volume_log'] = np.log(df['tick_volume'] + 1)
    return df

def scale_features(df, feature_columns):
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[feature_columns])
    return scaled_features, scaler

def prepare_transformer_data(df, lookback, forecast, sampling_rate, threshold, baseline_interval, log_func):
    X, y_direction = [], []
    close_values = df['close'].values  # Ensure 'close' is indexed correctly
    feature_columns = ['open', 'high', 'low', 'close', 'tick_volume_log', 'high_low', 'close_open']

    # Initialize the tqdm progress bar
    pbar = tqdm(range(lookback, len(df) - forecast), desc='Preparing data', unit='step')

    for i in pbar:
        window_data = df.iloc[i - lookback:i].copy()

        # Reset the baseline every baseline_interval
        if i % baseline_interval == 0:
            baseline_close = close_values[i]

        # Calculate the max and min close prices within the forecast range
        max_close = np.max(close_values[i:i + forecast])
        min_close = np.min(close_values[i:i + forecast])

        # Determine the rise, consolidate, or fall based on the differences
        if baseline_close != 0:
            max_diff = (max_close - baseline_close) / baseline_close
            min_diff = (baseline_close - min_close) / baseline_close
        else:
            max_diff = 0  # Handle case where baseline_close is zero
            min_diff = 0

        # Determine direction label: 0 = Fall, 1 = Consolidate, 2 = Rise
        if max_diff > threshold and max_diff > min_diff or min_diff < 0 and max_diff > threshold:
            direction_label = 2  # Rise
        elif min_diff > threshold and min_diff > max_diff or max_diff < 0 and min_diff > threshold:
            direction_label = 0  # Fall
        else:
            direction_label = 1  # Consolidate

        y_direction.append(direction_label)

        # Scale the features within the window
        scaler = MinMaxScaler()
        scaled_window_data = scaler.fit_transform(window_data[feature_columns])
        window_data[feature_columns] = scaled_window_data

        # Apply Fourier Transform to the window
        fft_values = fft(window_data['close'].values)
        magnitudes = np.abs(fft_values)
        frequencies = fftfreq(lookback, d=1 / sampling_rate)

        # Combine the original features with Fourier Transform features
        combined_features = np.hstack([window_data.values, magnitudes.reshape(-1, 1), frequencies.reshape(-1, 1)])
        X.append(combined_features)

    X, y_direction = np.array(X), np.array(y_direction)
    log_func("Data preparation completed.")
    return X, y_direction

def custom_loss_and_accuracy(y_true, y_pred):
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

    # Calculate accuracy based on the predicted percentages
    accuracy = rise_fall_pct

    # Return the base loss plus a penalty to guide the model and the accuracy
    return K.mean(base_loss)*0.01 + (1 - penalty) , accuracy

# Wrapper function to use custom loss in the model
def custom_loss(y_true, y_pred):
    loss, _ = custom_loss_and_accuracy(y_true, y_pred)
    return loss

# Wrapper function to use custom accuracy in the model
def custom_accuracy(y_true, y_pred):
    _, accuracy = custom_loss_and_accuracy(y_true, y_pred)
    return accuracy

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res

def build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    outputs_direction = Dense(3, activation="softmax", name="direction_output")(x)  # 3 categories: fall, consolidate, rise
    return Model(inputs, outputs_direction)

def plot_pie_chart(labels, sizes, title):
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.setp(autotexts, size=10, weight="bold")
    ax.set_title(title)
    plt.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.show()

class CustomLogger(Callback):
    def __init__(self, log_func):
        super().__init__()
        self.log_func = log_func

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        log_str = f'Epoch {epoch + 1}: '
        for key, value in logs.items():
            log_str += f'{key} = {value}; '
        self.log_func(log_str)

def train_model(lookback, forecast, threshold, epochs, batch_size, baseline_interval, days, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout, mlp_dropout, use_custom_loss, log_func):
    log_func("Initializing MetaTrader 5...")
    initialize_mt5()

    start_date = datetime.now() - timedelta(days=days)
    end_date = datetime.now()
    symbol = "XAUUSD"
    timeframe = mt5.TIMEFRAME_M5

    log_func(f"Fetching data for {symbol} from {start_date} to {end_date}...")
    df = fetch_data(symbol, timeframe, start_date, end_date)
    if df is None or df.empty:
        log_func("No data fetched. Exiting.")
        return

    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.set_index('time')

    log_func("Adding features...")
    df = add_features(df)

    sampling_rate = 1 / (5 / 60)  # 1 sample per 5 minutes
    log_func("Preparing transformer data...")
    X, y_direction = prepare_transformer_data(df, lookback, forecast, sampling_rate, threshold, baseline_interval, log_func)

    log_func("Splitting data into training and validation sets...")
    X_train, X_val, y_train_direction, y_val_direction = train_test_split(
        X, y_direction, test_size=0.2, random_state=42
    )

    log_func("Building transformer model...")
    model = build_transformer_model(
        input_shape=(lookback, X.shape[2]),
        head_size=head_size,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_transformer_blocks=num_transformer_blocks,
        mlp_units=mlp_units,
        dropout=dropout,
        mlp_dropout=mlp_dropout
    )

    loss_function = custom_loss if use_custom_loss else 'sparse_categorical_crossentropy'

    log_func("Compiling model...")
    model.compile(
        optimizer='adam',
        loss=loss_function,
        metrics=['accuracy', custom_accuracy] if use_custom_loss else ['accuracy']
    )

    log_func("Starting model training...")
    model.fit(
        X_train,
        y_train_direction,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val_direction),
        callbacks=[CustomLogger(log_func)]
    )

    model.save('forex_trend_transformer_model.h5')
    log_func("Model saved to forex_trend_transformer_model.h5")

    log_func("Evaluating model on validation data...")
    eval_results = model.evaluate(X_val, y_val_direction, verbose=1)
    for name, value in zip(model.metrics_names, eval_results):
        log_func(f'{name}: {value}')

    total_actual = len(y_val_direction)
    actual_rise = np.sum(y_val_direction == 2)
    actual_fall = np.sum(y_val_direction == 0)
    actual_consolidate = total_actual - (actual_rise + actual_fall)
    plot_pie_chart(
        labels=['Actual Rise', 'Actual Consolidate', 'Actual Fall'],
        sizes=[actual_rise, actual_consolidate, actual_fall],
        title='Actual Rise, Consolidate, and Fall Percentages'
    )

    val_predictions = model.predict(X_val)
    val_direction_predictions = np.argmax(val_predictions, axis=1)

    correct_rise_predictions = np.sum((val_direction_predictions == 2) & (y_val_direction == 2))
    incorrect_rise_predictions = np.sum((val_direction_predictions == 2) & (y_val_direction != 2))
    correct_fall_predictions = np.sum((val_direction_predictions == 0) & (y_val_direction == 0))
    incorrect_fall_predictions = np.sum((val_direction_predictions == 0) & (y_val_direction != 0))
    correct_consolidate_predictions = np.sum((val_direction_predictions == 1) & (y_val_direction == 1))
    incorrect_consolidate_predictions = np.sum((val_direction_predictions == 1) & (y_val_direction != 1))

    plot_pie_chart(
        labels=['Correct Rise Predictions', 'Incorrect Rise Predictions', 'Correct Fall Predictions', 'Incorrect Fall Predictions', 'Correct Consolidate Predictions', 'Incorrect Consolidate Predictions'],
        sizes=[correct_rise_predictions, incorrect_rise_predictions, correct_fall_predictions, incorrect_fall_predictions, correct_consolidate_predictions, incorrect_consolidate_predictions],
        title='Prediction Accuracy for Rise, Consolidate, and Fall'
    )

def main():
    def start_training():
        lookback = int(lookback_entry.get())
        forecast = int(forecast_entry.get())
        threshold = float(threshold_entry.get())
        epochs = int(epochs_entry.get())
        batch_size = int(batch_size_entry.get())
        baseline_interval = int(baseline_interval_entry.get())
        days = int(days_entry.get())
        head_size = int(head_size_entry.get())
        num_heads = int(num_heads_entry.get())
        ff_dim = int(ff_dim_entry.get())
        num_transformer_blocks = int(num_transformer_blocks_entry.get())
        mlp_units = [int(x) for x in mlp_units_entry.get().split(',')]
        dropout = float(dropout_entry.get())
        mlp_dropout = float(mlp_dropout_entry.get())
        use_custom_loss = custom_loss_var.get()

        log_area.delete('1.0', tk.END)  # Clear the log area
        train_model(lookback, forecast, threshold, epochs, batch_size, baseline_interval, days, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout, mlp_dropout, use_custom_loss, log_func)

    def log_func(message):
        log_area.insert(tk.END, message + '\n')
        log_area.see(tk.END)

    root = tk.Tk()
    root.title("Forex Model Training")

    tk.Label(root, text="LOOKBACK").grid(row=0, column=0)
    lookback_entry = tk.Entry(root)
    lookback_entry.grid(row=0, column=1)
    lookback_entry.insert(0, "288")

    tk.Label(root, text="FORECAST").grid(row=1, column=0)
    forecast_entry = tk.Entry(root)
    forecast_entry.grid(row=1, column=1)
    forecast_entry.insert(0, "144")

    tk.Label(root, text="THRESHOLD").grid(row=2, column=0)
    threshold_entry = tk.Entry(root)
    threshold_entry.grid(row=2, column=1)
    threshold_entry.insert(0, "0.004")

    tk.Label(root, text="EPOCHS").grid(row=3, column=0)
    epochs_entry = tk.Entry(root)
    epochs_entry.grid(row=3, column=1)
    epochs_entry.insert(0, "60")

    tk.Label(root, text="BATCH_SIZE").grid(row=4, column=0)
    batch_size_entry = tk.Entry(root)
    batch_size_entry.grid(row=4, column=1)
    batch_size_entry.insert(0, "64")

    tk.Label(root, text="BASELINE_INTERVAL").grid(row=5, column=0)
    baseline_interval_entry = tk.Entry(root)
    baseline_interval_entry.grid(row=5, column=1)
    baseline_interval_entry.insert(0, "3")

    tk.Label(root, text="DAYS").grid(row=6, column=0)
    days_entry = tk.Entry(root)
    days_entry.grid(row=6, column=1)
    days_entry.insert(0, "30")

    tk.Label(root, text="HEAD_SIZE").grid(row=7, column=0)
    head_size_entry = tk.Entry(root)
    head_size_entry.grid(row=7, column=1)
    head_size_entry.insert(0, "256")

    tk.Label(root, text="NUM_HEADS").grid(row=8, column=0)
    num_heads_entry = tk.Entry(root)
    num_heads_entry.grid(row=8, column=1)
    num_heads_entry.insert(0, "4")

    tk.Label(root, text="FF_DIM").grid(row=9, column=0)
    ff_dim_entry = tk.Entry(root)
    ff_dim_entry.grid(row=9, column=1)
    ff_dim_entry.insert(0, "4")

    tk.Label(root, text="NUM_TRANSFORMER_BLOCKS").grid(row=10, column=0)
    num_transformer_blocks_entry = tk.Entry(root)
    num_transformer_blocks_entry.grid(row=10, column=1)
    num_transformer_blocks_entry.insert(0, "4")

    tk.Label(root, text="MLP_UNITS (comma separated)").grid(row=11, column=0)
    mlp_units_entry = tk.Entry(root)
    mlp_units_entry.grid(row=11, column=1)
    mlp_units_entry.insert(0, "256,128,64")

    tk.Label(root, text="DROPOUT").grid(row=12, column=0)
    dropout_entry = tk.Entry(root)
    dropout_entry.grid(row=12, column=1)
    dropout_entry.insert(0, "0.3")

    tk.Label(root, text="MLP_DROPOUT").grid(row=13, column=0)
    mlp_dropout_entry = tk.Entry(root)
    mlp_dropout_entry.grid(row=13, column=1)
    mlp_dropout_entry.insert(0, "0.4")

    custom_loss_var = tk.BooleanVar()
    custom_loss_checkbox = tk.Checkbutton(root, text="Use Custom Loss", variable=custom_loss_var)
    custom_loss_checkbox.grid(row=14, columnspan=2)

    train_button = tk.Button(root, text="Start Training", command=start_training)
    train_button.grid(row=15, columnspan=2)

    log_area = scrolledtext.ScrolledText(root, width=80, height=20)
    log_area.grid(row=16, columnspan=2)

    root.mainloop()

if __name__ == "__main__":
    main()
