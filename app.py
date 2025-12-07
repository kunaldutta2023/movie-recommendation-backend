from flask import Flask, request, jsonify
from flask_cors import CORS
from model_loader import recommend

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({"message": "Movie Recommendation API is running!"})

@app.route('/recommend', methods=['GET'])
def get_recommendation():
    movie_name = request.args.get("movie", "")

    if not movie_name:
        return jsonify({"error": "Movie name is required"}), 400

    results = recommend(movie_name)

    return jsonify({"movies": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
