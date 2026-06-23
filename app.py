from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, LSTM
from tensorflow.keras.utils import to_categorical
import threading
import os

app = Flask(__name__)
CORS(app)

models_ready = False
rnn_model = None
lstm_model = None
scaler = None
label_encoder = None
rnn_accuracy = None
lstm_accuracy = None
training_status = "idle"   # idle | training | ready | error
training_error = ""

SELECTED_FEATURES = [
    'Temperature_C',
    'Humidity_Percent',
    'Pressure_hPa',
    'Wind_Speed_mps',
    'Precipitation_mm',
    'Cloud_Cover_Percent',
    'UV_Index'
]
TARGET_COLUMN = 'Weather_Class'


def train_models():
    global rnn_model, lstm_model, scaler, label_encoder
    global rnn_accuracy, lstm_accuracy, models_ready, training_status, training_error

    try:
        training_status = "training"

        data = pd.read_csv("weather_dataset.csv")
        data = data[SELECTED_FEATURES + [TARGET_COLUMN]]

        label_encoder = LabelEncoder()
        data[TARGET_COLUMN] = label_encoder.fit_transform(data[TARGET_COLUMN])

        X = data[SELECTED_FEATURES].values
        y = data[TARGET_COLUMN].values

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        num_classes = len(np.unique(y))
        y_train_cat = to_categorical(y_train, num_classes)
        y_test_cat  = to_categorical(y_test,  num_classes)

        X_train_seq = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test_seq  = X_test.reshape(X_test.shape[0],  1, X_test.shape[1])

        rnn_model = Sequential([
            SimpleRNN(128, activation='relu', input_shape=(1, X_train.shape[1])),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        rnn_model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        rnn_model.fit(X_train_seq, y_train_cat,
                      epochs=20, batch_size=32,
                      validation_split=0.2, verbose=1)
        _, rnn_acc = rnn_model.evaluate(X_test_seq, y_test_cat, verbose=0)
        rnn_accuracy = float(rnn_acc)

        lstm_model = Sequential([
            LSTM(128, activation='tanh', input_shape=(1, X_train.shape[1])),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        lstm_model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        lstm_model.fit(X_train_seq, y_train_cat,
                       epochs=20, batch_size=32,
                       validation_split=0.2, verbose=1)
        _, lstm_acc = lstm_model.evaluate(X_test_seq, y_test_cat, verbose=0)
        lstm_accuracy = float(lstm_acc)

        rnn_model.save("rnn_climate_model.keras")
        lstm_model.save("lstm_climate_model.keras")

        models_ready    = True
        training_status = "ready"
        print(f"\n Training complete — RNN: {rnn_accuracy:.4f}  LSTM: {lstm_accuracy:.4f}")

    except Exception as e:
        training_status = "error"
        training_error  = str(e)
        print(f" Training error: {e}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "status":        training_status,
        "models_ready":  models_ready,
        "rnn_accuracy":  round(rnn_accuracy  * 100, 2) if rnn_accuracy  is not None else None,
        "lstm_accuracy": round(lstm_accuracy * 100, 2) if lstm_accuracy is not None else None,
        "error":         training_error
    })


@app.route("/train", methods=["POST"])
def train():
    global training_status
    if training_status == "training":
        return jsonify({"message": "Training already in progress."}), 400
    if training_status == "ready":
        return jsonify({"message": "Models already trained."}), 400

    thread = threading.Thread(target=train_models, daemon=True)
    thread.start()
    return jsonify({"message": "Training started."})


@app.route("/predict", methods=["POST"])
def predict():
    if not models_ready:
        return jsonify({"error": "Models not trained yet."}), 503

    body = request.get_json(force=True)
    try:
        user_input = np.array([[
            float(body["temperature"]),
            float(body["humidity"]),
            float(body["pressure"]),
            float(body["wind_speed"]),
            float(body["precipitation"]),
            float(body["cloud_cover"]),
            float(body["uv_index"])
        ]])
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    scaled = scaler.transform(user_input)
    seq    = scaled.reshape(1, 1, 7)

    rnn_pred      = rnn_model.predict(seq, verbose=0)
    rnn_class     = int(np.argmax(rnn_pred))
    rnn_label     = label_encoder.inverse_transform([rnn_class])[0]
    rnn_conf      = float(np.max(rnn_pred)) * 100
    lstm_pred     = lstm_model.predict(seq, verbose=0)
    lstm_class    = int(np.argmax(lstm_pred))
    lstm_label    = label_encoder.inverse_transform([lstm_class])[0]
    lstm_conf     = float(np.max(lstm_pred)) * 100

    best = "LSTM" if lstm_accuracy >= rnn_accuracy else "RNN"

    return jsonify({
        "rnn": {
            "label":      rnn_label,
            "confidence": round(rnn_conf, 1)
        },
        "lstm": {
            "label":      lstm_label,
            "confidence": round(lstm_conf, 1)
        },
        "best_model":    best,
        "rnn_accuracy":  round(rnn_accuracy  * 100, 2),
        "lstm_accuracy": round(lstm_accuracy * 100, 2)
    })

if __name__ == "__main__":
    # Auto-train on startup if dataset exists
    if os.path.exists("weather_dataset.csv"):
        print(" Dataset found — starting training automatically...")
        thread = threading.Thread(target=train_models, daemon=True)
        thread.start()
    else:
        print(" weather_dataset.csv not found. POST /train after placing the file.")

    app.run(debug=False, port=5000)