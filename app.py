from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import pandas as pd
import joblib
import os
import datetime

app = Flask(__name__, static_folder="frontend")
CORS(app)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///transactions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Transaction Model
class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    amount = db.Column(db.Float, nullable=False)
    location = db.Column(db.String(100), nullable=False)
    merchant_type = db.Column(db.String(100), nullable=False)
    device_type = db.Column(db.String(50), nullable=False)
    time_of_transaction = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    risk_score = db.Column(db.Integer, nullable=False)
    prediction = db.Column(db.String(50), nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'amount': self.amount,
            'location': self.location,
            'merchant_type': self.merchant_type,
            'device_type': self.device_type,
            'time_of_transaction': self.time_of_transaction,
            'timestamp': self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            'risk_score': self.risk_score,
            'prediction': self.prediction
        }

# Load models on startup
try:
    model = joblib.load('models/isolation_forest_model.pkl')
    preprocessor = joblib.load('models/preprocessor.pkl')
    print("Models loaded successfully.")
except FileNotFoundError:
    print("Warning: Model files not found. Please run train_model.py first.")
    model, preprocessor = None, None

@app.route("/")
def home():
    return send_from_directory("frontend", "index.html")

@app.route("/check", methods=["POST"])
def check():
    if not model or not preprocessor:
        return jsonify({"result": "Error: Models not loaded", "score": 0}), 500

    data = request.json

    try:
        amount = float(data.get("amount", 0))
        location = data.get("location", "Other")
        merchant_type = data.get("merchant_type", "Other")
        device_type = data.get("device_type", "Unknown")
        time_of_transaction = int(data.get("time", datetime.datetime.now().hour))

        input_data = pd.DataFrame([{
            'amount': amount,
            'location': location,
            'merchant_type': merchant_type,
            'device_type': device_type,
            'time_of_transaction': time_of_transaction
        }])

        processed_data = preprocessor.transform(input_data)

        prediction_val = model.predict(processed_data)[0]
        score_raw = model.decision_function(processed_data)[0]

        risk_score = max(0, min(100, int((0.5 - score_raw) * 100)))

        if prediction_val == -1:
            result_text = "⚠️ SUSPICIOUS TRANSACTION DETECTED"
            prediction_label = "SUSPICIOUS"
        else:
            result_text = "✅ NORMAL TRANSACTION"
            prediction_label = "NORMAL"

        # Save to database
        new_transaction = Transaction(
            amount=amount,
            location=location,
            merchant_type=merchant_type,
            device_type=device_type,
            time_of_transaction=time_of_transaction,
            risk_score=risk_score,
            prediction=prediction_label
        )
        db.session.add(new_transaction)
        db.session.commit()

        return jsonify({
            "result": result_text,
            "score": risk_score,
            "prediction": prediction_label
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"result": f"Error: {str(e)}", "score": 0}), 400

@app.route("/history", methods=["GET"])
def history():
    try:
        transactions = Transaction.query.order_by(Transaction.timestamp.desc()).limit(10).all()
        return jsonify([t.to_dict() for t in transactions])
    except Exception as e:
        print(f"Error fetching history: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=5000, debug=True)
