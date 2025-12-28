from flask import Flask, render_template, request
from pipeline.prediction_pipeline import hybrid_recommendation

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = None
    error = None

    if request.method == 'POST':
        try:
            user_id = int(request.form.get("userID"))
            recommendations = hybrid_recommendation(user_id)
        except Exception as e:
            error = "An error occurred while generating recommendations."
            print(f"Error occurred: {e}")

    return render_template("index.html",
                           recommendations=recommendations,
                           error=error)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
