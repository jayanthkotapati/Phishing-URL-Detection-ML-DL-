"""Flask web app for the phishing URL detection learning project."""
from pathlib import Path
import io, os, sqlite3
from functools import wraps
import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from feature_extraction import features_for_urls

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "phishing_pipeline.joblib"
DB_PATH = BASE_DIR / "instance" / "users.sqlite3"
UPLOAD_DIR = BASE_DIR / "Uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "local-demo-change-this-secret")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB upload limit


def get_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL)")


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


def load_pipeline():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)["pipeline"]

@app.route("/")
def index():
    return redirect(url_for("dashboard" if "user_id" in session else "login"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not username or not email or len(password) < 8:
            flash("Enter a username, email, and password of at least 8 characters.", "danger")
            return render_template("register.html")
        try:
            with get_db() as conn:
                cur = conn.execute("INSERT INTO users(username,email,password_hash) VALUES(?,?,?)", (username, email, generate_password_hash(password)))
                user_id = cur.lastrowid
            session.update(user_id=user_id, username=username)
            return redirect(url_for("dashboard"))
        except sqlite3.IntegrityError:
            flash("That username or email is already registered.", "danger")
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        identity = request.form.get("identity", "").strip()
        password = request.form.get("password", "")
        with get_db() as conn:
            user = conn.execute("SELECT * FROM users WHERE username=? OR email=?", (identity, identity.lower())).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session.update(user_id=user["id"], username=user["username"])
            return redirect(url_for("dashboard"))
        flash("Invalid username/email or password.", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/dashboard")
@login_required
def dashboard():
    ready = MODEL_PATH.exists()
    return render_template("dashboard.html", model_ready=ready)

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    result = None
    submitted_url = ""
    if request.method == "POST":
        submitted_url = request.form.get("url", "").strip()
        if not submitted_url or " " in submitted_url:
            flash("Enter one URL without spaces.", "danger")
        else:
            pipeline = load_pipeline()
            if pipeline is None:
                flash("Model not found. Run 'python train_model.py' first.", "warning")
            else:
                X = features_for_urls([submitted_url])
                label = int(pipeline.predict(X)[0])
                probability = pipeline.predict_proba(X)[0]
                result = {"label": "Phishing" if label else "Legitimate", "confidence": round(float(probability[label]) * 100, 1)}
    return render_template("predict.html", result=result, submitted_url=submitted_url)

@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    results_html = None
    if request.method == "POST":
        file = request.files.get("file")
        if not file or not file.filename.lower().endswith(".csv"):
            flash("Choose a CSV file with a column named 'url'.", "danger")
        else:
            try:
                data = pd.read_csv(file)
                if "url" not in data.columns:
                    raise ValueError("CSV must contain a column named 'url'.")
                if len(data) > 5000:
                    raise ValueError("Upload no more than 5,000 URLs at a time.")
                pipeline = load_pipeline()
                if pipeline is None:
                    raise ValueError("Model not found. Run 'python train_model.py' first.")
                urls = data["url"].fillna("").astype(str).tolist()
                valid = [u.strip() for u in urls]
                X = features_for_urls(valid)
                preds = pipeline.predict(X)
                probs = pipeline.predict_proba(X)
                data["prediction"] = ["Phishing" if int(p) else "Legitimate" for p in preds]
                data["confidence_percent"] = [round(float(probs[i][int(preds[i])]) * 100, 1) for i in range(len(preds))]
                output = io.BytesIO(data.to_csv(index=False).encode("utf-8"))
                output.seek(0)
                session["last_results"] = data.to_dict(orient="records")
                results_html = data.head(100).to_html(index=False, classes="data-table", border=0)
                flash(f"Processed {len(data)} URLs. Results table shows up to 100 rows.", "success")
            except Exception as exc:
                flash(f"Could not process CSV: {exc}", "danger")
    return render_template("upload.html", results_html=results_html)

@app.route("/download-results")
@login_required
def download_results():
    rows = session.get("last_results")
    if not rows:
        flash("Upload a CSV first to create results.", "warning")
        return redirect(url_for("upload"))
    out = io.BytesIO(pd.DataFrame(rows).to_csv(index=False).encode("utf-8"))
    out.seek(0)
    return send_file(out, mimetype="text/csv", as_attachment=True, download_name="phishing_predictions.csv")

@app.route("/metrics")
@login_required
def metrics():
    path = BASE_DIR / "models" / "metrics.json"
    data = None
    if path.exists():
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
    return render_template("metrics.html", metrics=data)

@app.errorhandler(413)
def too_large(_error):
    flash("File too large. Maximum upload size is 5 MB.", "danger")
    return redirect(url_for("upload"))

init_db()
if __name__ == "__main__":
    app.run(debug=True)
