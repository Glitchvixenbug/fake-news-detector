
from flask import Flask, request, render_template_string
import joblib
import os

vectorizer = joblib.load('tfidf_vectorizer.joblib')
model = joblib.load('logreg_model.joblib')

app = Flask(__name__)

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Fake News Detector</title>
</head>
<body>
    <h2>Fake News Detector</h2>
    <form method="post">
        <textarea name="news_text" rows="8" cols="80" placeholder="Paste news article here"></textarea><br>
        <input type="submit" value="Check News">
    </form>
    {% if prediction is not none %}
        <h3>Prediction: {{ prediction }}</h3>
        <p>Probability FAKE: {{ prob }}</p>
    {% endif %}
</body>
</html>
'''

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
