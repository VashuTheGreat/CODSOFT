from flask import Flask, render_template, request, jsonify
import pickle
from model import MoviePredictorModel
from main import FetchData
import json
app = Flask(__name__)

# Loading pkl
similarities = pickle.load(open('similarities.pkl', 'rb'))
new_movie = pickle.load(open('new_movie.pkl', 'rb'))
mov_name = pickle.load(open('names.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('index.html', mov_name=mov_name)

@app.route("/predict", methods=['POST'])
def predict():
    data = request.get_json()  # Geting JSON Data by fetching

    movie_name = data.get('name')
    num = int(data.get('num'))

    model = MoviePredictorModel()
    model.compile(new_movie)
    model.fit(similarities)

    pred = model.predict(movie_name, num)
    fet=FetchData()
    print(pred)
    movies_to_show = [fet.fetch(i[0]) for i in pred]

    return jsonify({'prediction': movies_to_show})


@app.route('/page', methods=['POST'])
def page():
    movie_data = json.loads(request.form['movie'])  
    return render_template('page.html', movie=movie_data)

if __name__ == "__main__":
    app.run(debug=True)
