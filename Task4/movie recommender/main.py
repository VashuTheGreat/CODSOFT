import requests
from dotenv import load_dotenv
import os

load_dotenv()  

api_key = os.getenv("API")

class FetchData:
    def fetch(self, movie_id):
        url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US'

        response = requests.get(url)

        if response.status_code != 200:
            print(f"Error fetching movie {movie_id}: {response.status_code}")
            return {"error": f"Failed to fetch data. Status code: {response.status_code}"}

        data = response.json()

        return {
            "title": data.get('original_title'),
            "genres": [x['name'] for x in data.get('genres', [])],
            "poster": "https://image.tmdb.org/t/p/w500/" + data.get('poster_path', ''),
            "homepage":data['homepage'],
            "release_date": data.get('release_date'),
            "original_language": data.get('original_language'),
            "spoken_language": [x['name'] for x in data.get('spoken_languages', [])],
            "popularity": data.get('popularity'),
            "production_companies": [x['name'] for x in data.get('production_companies', [])],
            "production_countries": [x['name'] for x in data.get('production_countries', [])],
            "revenue": data.get('revenue'),
            "runtime": data.get('runtime'),
            "tagline": data.get('tagline')
        }
