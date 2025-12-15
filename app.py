import os, json, pickle,requests
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_from_directory
from flask_cors import CORS
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
from models import init_db, SessionLocal, User, SessionHistory
from sqlalchemy.orm import scoped_session
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime


# -------------------------
#   Database session
# -------------------------
db_session = scoped_session(SessionLocal)

# -------------------------
#   Flask App Setup
# -------------------------
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

app.secret_key = "super_secret_local_key"
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)



# -------------------------
#   Load ML Artifacts
MODEL_URL = "https://huggingface.co/Kartikey27/ai-medical-diagnosis-model/resolve/main/model.pkl"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

model = None

def load_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            print("⬇ Downloading model from Hugging Face...")
            with requests.get(MODEL_URL, stream=True) as r:
                r.raise_for_status()
                with open(MODEL_PATH, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

        print("✅ Loading model into memory...")
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

    return model 

symptom_columns = json.load(open(os.path.join(MODEL_DIR, "symptom_columns.json"), "r"))
medical_data = json.load(open(os.path.join(MODEL_DIR, "medical_data.json"), "r"))
disease_map = json.load(open(os.path.join(MODEL_DIR, "disease_mapping.json"), "r"))

# -------------------------
#   Initialize DB
# -------------------------
init_db()

# -------------------------
#   Routes
# -------------------------
@app.route("/")
def index():
    return render_template("index.html", symptom_columns=symptom_columns, user=session.get("user_id"))


@app.route("/login")
def login_page():
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


@app.route("/register")
def register_page():
    return render_template("register.html")

@app.route("/history")
def history():
    if "user_id" not in session:
        return redirect("/login")

    user_id = session["user_id"]
    history_items = db_session.query(SessionHistory).filter_by(user_id=user_id).all()

    return render_template("history.html", history=history_items)

# -------------------------
#   Register User
# -------------------------
@app.route("/register_user", methods=["POST"])
def register_user():
    data = request.form
    email = data["email"]
    password = generate_password_hash(data["password"])

    # check existing
    if db_session.query(User).filter(User.email == email).first():
        return "User already exists!"

    new_user = User(email=email, password=password)
    db_session.add(new_user)
    db_session.commit()

    return redirect(url_for("login_page"))

# -------------------------
#   Login User
# -------------------------
@app.route("/login_user", methods=["POST"])
def login_user():
    data = request.form
    email = data["email"]
    password = data["password"]

    user = db_session.query(User).filter(User.email == email).first()
    if not user or not check_password_hash(user.password, password):
        return "Invalid credentials!"

    session["user_id"] = user.id
    return redirect(url_for("index"))

# -------------------------
#   Prediction API
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "user_id" not in session:
        return jsonify({"error": "login required"}), 401
    model = load_model()

    data = request.json
    symptoms = data["symptoms"]

    vector = [1 if s in symptoms else 0 for s in symptom_columns]
    pred_raw = model.predict(np.array(vector).reshape(1, -1))[0]

    disease = disease_map.get(str(pred_raw), "Unknown")

    # get disease details
    details = medical_data.get(disease, {})
    precautions = details.get("precautions", [])
    medications = details.get("medications", [])
    diet = details.get("diet", [])
    workout = details.get("workout", [])

    # SAVE HISTORY (without report_file for now)
    try:
        record = SessionHistory(
            user_id=session["user_id"],
            disease=disease,
            symptoms=",".join(symptoms),
            timestamp=datetime.now()
        )
        db_session.add(record)
        db_session.commit()
    except Exception as e:
        print("Failed to save history:", e)

    return jsonify({
        "disease": disease,
        "precautions": precautions,
        "medications": medications,
        "diet": diet,
        "workout": workout
    })

# -------------------------
#   PDF Download
@app.route("/generate_pdf", methods=["POST"])
def generate_pdf():
    try:
        data = request.json
        disease = data.get("disease", "Unknown")
        precautions = data.get("precautions", [])
        medications = data.get("medications", [])
        diet = data.get("diet", [])
        workout = data.get("workout", [])

        filename = f"report_{int(datetime.now().timestamp())}.pdf"
        path = os.path.join(REPORT_DIR, filename)

        c = canvas.Canvas(path, pagesize=letter)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, 750, "AI Medical Diagnosis Report")

        c.setFont("Helvetica", 12)
        c.drawString(50, 720, f"Disease: {disease}")

        y = 690

        def write_section(title, items):
            nonlocal y
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y, title)
            y -= 20

            c.setFont("Helvetica", 11)
            for item in items:
                c.drawString(70, y, "- " + item)
                y -= 15
            y -= 10

        write_section("Precautions", precautions)
        write_section("Medications", medications)
        write_section("Diet Advice", diet)
        write_section("Workout / Lifestyle", workout)

        c.save()

        # return filename so frontend can trigger download
        return jsonify({"file": filename})

    except Exception as e:
        print("PDF ERROR:", e)
        return jsonify({"error": str(e)})


@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(REPORT_DIR, filename, as_attachment=True)

# -------------------------
#   Run App
# -------------------------
@app.teardown_appcontext
def shutdown_session(exception=None):
    db_session.remove()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

