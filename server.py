import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs
import json
import pandas as pd
from uuid import uuid4
from datetime import datetime
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

# Download NLTK resources
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize sentiment analyzer and stop words
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# Allowed locations
ALLOWED_LOCATIONS = [
    "Albuquerque, New Mexico", "Carlsbad, California", "Chula Vista, California", "Colorado Springs, Colorado",
    "Denver, Colorado", "El Cajon, California", "El Paso, Texas", "Escondido, California", "Fresno, California",
    "La Mesa, California", "Las Vegas, Nevada", "Los Angeles, California", "Oceanside, California",
    "Phoenix, Arizona", "Sacramento, California", "Salt Lake City, Utah", "San Diego, California", "Tucson, Arizona"
]

# Load reviews from CSV
reviews = pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        pass

    def analyze_sentiment(self, review_body: str) -> dict:
        # Analyze the sentiment of a given review body using Vader
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        # Handling GET requests
        if environ["REQUEST_METHOD"] == "GET":
            query_string = environ.get("QUERY_STRING", "")
            query_params = parse_qs(query_string)

            location = query_params.get("location", [None])[0]
            start_date = query_params.get("start_date", [None])[0]
            end_date = query_params.get("end_date", [None])[0]

            filtered_reviews = reviews

            if location:
                filtered_reviews = [review for review in filtered_reviews if review['Location'] == location]

            if start_date:
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
                filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review['Timestamp'], "%Y-%m-%d %H:%M:%S") >= start_date]

            if end_date:
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
                filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review['Timestamp'], "%Y-%m-%d %H:%M:%S") <= end_date]

            for review in filtered_reviews:
                if 'sentiment' not in review:
                    review['sentiment'] = self.analyze_sentiment(review['ReviewBody'])

            sorted_reviews = sorted(filtered_reviews, key=lambda r: r.get('sentiment', {}).get('compound', 0), reverse=True)
            response_body = json.dumps(sorted_reviews, indent=2).encode("utf-8")

            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            return [response_body]

        # Handle POST requests
        if environ["REQUEST_METHOD"] == "POST":
            try:
                request_size = int(environ.get("CONTENT_LENGTH", 0))
            except ValueError:
                request_size = 0

            request_body = environ["wsgi.input"].read(request_size).decode("utf-8")
            post_data = parse_qs(request_body)

            review_body = post_data.get('ReviewBody', [None])[0]
            location = post_data.get('Location', [None])[0]

            if not review_body or not location:
                response_body = json.dumps({'error': 'Both ReviewBody and Location are required'}).encode('utf-8')
                start_response("400 Bad Request", [("Content-Type", "application/json")])
                return [response_body]

            if location not in ALLOWED_LOCATIONS:
                response_body = json.dumps({'error': 'Invalid location'}).encode('utf-8')
                start_response("400 Bad Request", [("Content-Type", "application/json")])
                return [response_body]

            new_review = {
                'ReviewId': str(uuid4()),
                'ReviewBody': review_body,
                'Location': location,
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'sentiment': self.analyze_sentiment(review_body)
            }

            reviews.append(new_review)
            response_body = json.dumps(new_review).encode("utf-8")
            start_response("201 Created", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            return [response_body]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = int(os.environ.get('PORT', 8000))  # Ensure port is an integer
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()
