import os, json, pickle, requests
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, send_from_directory
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.orm import scoped_session
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from models import init_db, SessionLocal, User, SessionHistory

# -------------------------
# Database
# -------------------------
db_session = scoped_session(SessionLocal)

# -------------------------
# Flask App
# -------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# -------------------------
# Load ML Model
# -------------------------
MODEL_URL = "https://huggingface.co/Kartikey27/ai-medical-diagnosis-model/resolve/main/model.pkl"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
model = None

def load_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            with requests.get(MODEL_URL, stream=True) as r:
                r.raise_for_status()
                with open(MODEL_PATH, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    return model

symptom_columns = json.load(open(os.path.join(MODEL_DIR, "symptom_columns.json")))
medical_data = json.load(open(os.path.join(MODEL_DIR, "medical_data.json")))
disease_map = json.load(open(os.path.join(MODEL_DIR, "disease_mapping.json")))

init_db()

# -------------------------
# Pages
# -------------------------
@app.route("/")
def index():
    return render_template("index.html", symptom_columns=symptom_columns)

@app.route("/login")
def login_page():
    return render_template("login.html")

@app.route("/register")
def register_page():
    return render_template("register.html")

# -------------------------
# Auth APIs
# -------------------------
@app.route("/register_user", methods=["POST"])
def register_user():
    email = request.form["email"]
    password = generate_password_hash(request.form["password"])

    if db_session.query(User).filter_by(email=email).first():
        return "User already exists", 400

    user = User(email=email, password=password)
    db_session.add(user)
    db_session.commit()
    return redirect("/login")

@app.route("/login_user", methods=["POST"])
def login_user():
    email = request.form["email"]
    password = request.form["password"]

    user = db_session.query(User).filter_by(email=email).first()
    if not user or not check_password_hash(user.password, password):
        return jsonify({"error": "Invalid credentials"}), 401

    # TOKEN = user.id (simple & enough for project)
    return jsonify({"token": str(user.id)})

# -------------------------
# Prediction API (TOKEN BASED)
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    token = request.headers.get("Authorization")
    if not token:
        return jsonify({"error": "login required"}), 401

    user_id = int(token)
    model = load_model()

    data = request.json
    symptoms = data["symptoms"]

    vector = [1 if s in symptoms else 0 for s in symptom_columns]
    pred = model.predict(np.array(vector).reshape(1, -1))[0]
    disease = disease_map.get(str(pred), "Unknown")

    details = medical_data.get(disease, {})
    precautions = details.get("precautions", [])
    medications = details.get("medications", [])
    diet = details.get("diet", [])
    workout = details.get("workout", [])

    record = SessionHistory(
        user_id=user_id,
        disease=disease,
        symptoms=",".join(symptoms),
        timestamp=datetime.now()
    )
    db_session.add(record)
    db_session.commit()

    return jsonify({
        "disease": disease,
        "precautions": precautions,
        "medications": medications,
        "diet": diet,
        "workout": workout
    })

# -------------------------
# PDF
# -------------------------
@app.route("/generate_pdf", methods=["POST"])
def generate_pdf():
    data = request.json
    filename = f"report_{int(datetime.now().timestamp())}.pdf"
    path = os.path.join(REPORT_DIR, filename)

    c = canvas.Canvas(path, pagesize=letter)
    y = 750

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "AI Medical Diagnosis Report")
    y -= 30

    c.setFont("Helvetica", 12)
    c.drawString(50, y, f"Disease: {data['disease']}")
    y -= 30

    for section in ["precautions", "medications", "diet", "workout"]:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, section.capitalize())
        y -= 20
        c.setFont("Helvetica", 11)
        for item in data.get(section, []):
            c.drawString(70, y, "- " + item)
            y -= 15
        y -= 10

    c.save()
    return jsonify({"file": filename})

@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(REPORT_DIR, filename, as_attachment=True)

@app.teardown_appcontext
def shutdown_session(exception=None):
    db_session.remove()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
