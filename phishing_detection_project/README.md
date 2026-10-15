# Enhancing Phishing Detection: Machine Learning with Feature Selection

A runnable educational prototype based on the project report **“Enhancing Phishing Detection: A Machine Learning Approach with Feature Selection and Deep Learning Models.”** It demonstrates URL feature extraction, feature selection, Random Forest training, evaluation, URL prediction, CSV upload, and a small Flask web interface.

> **Important:** The included CSV is a small, illustrative dataset created for testing the project workflow. It is not a production-grade or representative phishing dataset, and the resulting model must not be used as a security product or as the sole basis for deciding whether a website is safe. Replace it with a properly sourced, licensed, labeled dataset and evaluate on a time-separated test set before drawing performance conclusions.

## Features
- Register and log in using a local SQLite database (passwords are hashed).
- Enter a URL and get a demo classification: `Phishing` or `Legitimate`.
- Upload a CSV containing a `url` column to classify multiple URLs.
- Train a Random Forest pipeline with `SelectKBest` feature selection.
- Evaluate with accuracy, precision, recall, F1-score, and a confusion matrix.
- Download CSV prediction results.

## Tech stack
Python, Flask, pandas, NumPy, scikit-learn, joblib, SQLite, HTML/CSS.

## Requirements
- Python 3.10 or newer
- Internet access only to install Python dependencies (the application itself runs locally)

## Run on Windows (PowerShell)
1. Extract the ZIP file.
2. Open the extracted `phishing_detection_project` folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Create a virtual environment:
   ```powershell
   py -m venv .venv
   ```
5. Activate it:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` in that terminal and activate again.
6. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
7. Train the demo model:
   ```powershell
   python train_model.py
   ```
8. Start the website:
   ```powershell
   python app.py
   ```
9. Open `http://127.0.0.1:5000` in your browser and register a local account.
10. Stop the server with `Ctrl+C` in the terminal.

## Run on macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
python app.py
```
Then open `http://127.0.0.1:5000`.

## Dataset format
`data/phishing_urls.csv` contains `url,label` columns. Labels are `phishing` or `legitimate`. To train on your own dataset, preserve these column names or update `train_model.py`. Use a trusted source and check its license. Avoid training and testing on duplicate or near-duplicate URLs.

## Project workflow
1. Collect labeled URL examples.
2. Extract structural features (URL length, dots, subdomains, HTTPS, IP address, suspicious tokens, and related signals).
3. Split the data into training and testing subsets.
4. Select features with `SelectKBest`.
5. Train the Random Forest classifier.
6. Evaluate it on held-out data.
7. Load the saved pipeline for URL prediction or CSV batch prediction.

## Repository layout
```text
phishing_detection_project/
├── app.py                 # Flask app, login, URL prediction and CSV upload
├── train_model.py         # Train and evaluate the model
├── feature_extraction.py  # URL-to-feature conversion
├── requirements.txt
├── README.md
├── data/phishing_urls.csv # Small illustrative demo data
├── models/                # Model artifact created by training
├── templates/             # HTML pages
└── static/style.css       # UI styling
```

## Limitations and security notes
- URL-only features cannot reliably detect every phishing website. A legitimate-looking URL may still host malicious content, and a suspicious-looking URL is not proof of phishing.
- The sample dataset is synthetic and intentionally small; metrics from it do not establish real-world performance.
- Change `SECRET_KEY` through an environment variable before any deployment.
- This prototype is for local learning. Do not expose it publicly without security review, rate limiting, HTTPS, CSRF protection, and production configuration.
- The report lists XGBoost and MLP as additional model approaches. This starter implementation uses Random Forest plus feature selection to keep setup lightweight; those models can be added for comparative experiments.
