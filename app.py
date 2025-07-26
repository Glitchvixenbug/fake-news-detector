from flask import Flask, request, render_template_string
import joblib
import os

# Load vectorizer and model
vectorizer = joblib.load('tfidf_vectorizer.joblib')
model = joblib.load('logreg_model.joblib')

app = Flask(__name__)

# Professional, mobile-responsive HTML template
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Fake News Detector</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <!-- Bootstrap 5 CSS CDN -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #e0e7ff 0%, #f9fafb 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 450px;
            margin-top: 50px;
            padding: 30px 30px 20px 30px;
            background: #fff;
            border-radius: 20px;
            box-shadow: 0 6px 24px rgba(0,0,0,0.11);
        }
        .btn-primary {
            background: #4f46e5;
            border: none;
            border-radius: 10px;
            font-weight: 600;
        }
        .result-card {
            border-radius: 12px;
            background: #e0e7ff;
            padding: 15px;
            margin-top: 25px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.07);
            text-align: center;
        }
        .fake { color: #b91c1c; font-weight: bold; }
        .real { color: #059669; font-weight: bold; }
        .logo {
            font-size: 2.1rem;
            font-weight: 800;
            color: #4f46e5;
            letter-spacing: 2px;
            margin-bottom: 18px;
            text-align: center;
        }
    </style>
</head>
<body>
<div class="container">
    <div class="logo">Fake News Detector</div>
    <form method="post">
        <label for="news_text" class="form-label">Paste news article below:</label>
        <textarea name="news_text" class="form-control mb-3" rows="7" required placeholder="Paste news article here..."></textarea>
        <button type="submit" class="btn btn-primary w-100">Check News</button>
    </form>
    {% if prediction is not none %}
    <div class="result-card mt-4">
        <div class="{{ 'fake' if prediction == 'FAKE' else 'real' }}">
            <span style="font-size:1.2rem;">Prediction: {{ prediction }}</span>
        </div>
        <div>
            Probability FAKE: <b>{{ prob }}</b>
        </div>
    </div>
    {% endif %}
    <footer class="text-center mt-4" style="font-size: 0.9rem;">
        &copy; 2025 Fake News Detector | Powered by AI
    </footer>
</div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    prob = None
    if request.method == 'POST':
        news_text = request.form['news_text']
        X = vectorizer.transform([news_text])
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]
        prediction = "FAKE" if pred == 1 else "REAL"
        prob = round(prob, 2)
    return render_template_string(HTML, prediction=prediction, prob=prob)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
