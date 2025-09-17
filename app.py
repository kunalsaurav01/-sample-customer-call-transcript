import os
import sys
import csv
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from groq import Groq

# ------------------ Configuration ------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Set in environment
USE_MOCK_MODE = False if GROQ_API_KEY else True  # If no key, run mock mode

# Initialize Flask app
app = Flask(__name__)

# Initialize Groq client if API key exists
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ------------------ Analysis Function ------------------

def analyze_with_groq(transcript):
    if not client:
        raise Exception("Groq client not initialized. Check your API key.")

    try:
        # Summary
        summary_response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert customer service analyst. Summarize the conversation in 2-3 clear sentences."},
                {"role": "user", "content": f"Summarize this customer call transcript:\n\n{transcript}"}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=150
        )
        summary = summary_response.choices[0].message.content.strip()

        # Sentiment
        sentiment_response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Respond with one word: Positive, Negative, or Neutral based on the customer's emotions."},
                {"role": "user", "content": f"Analyze sentiment in this transcript:\n\n{transcript}"}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=10
        )
        sentiment = sentiment_response.choices[0].message.content.strip()

        if sentiment.lower() not in ["positive", "negative", "neutral"]:
            sentiment = "Neutral"

        return summary, sentiment

    except Exception as e:
        raise Exception(f"Groq API error: {str(e)}")

# ------------------ Save to CSV ------------------
def save_to_csv(transcript, summary, sentiment):
    filename = 'call_analysis.csv'
    file_exists = os.path.isfile(filename)

    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Timestamp', 'Transcript', 'Summary', 'Sentiment']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Transcript': transcript.replace('\n', ' '),
            'Summary': summary,
            'Sentiment': sentiment
        })

# ------------------ Routes ------------------
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/analyze', methods=['POST'])
def analyze_transcript():
    data = request.get_json()
    transcript = data.get('transcript', '').strip()

    if not transcript or len(transcript) < 20:
        return jsonify({'error': 'Transcript too short or empty'}), 400

    try:
        summary, sentiment = analyze_with_groq(transcript)
        save_to_csv(transcript, summary, sentiment)

        return jsonify({
            'transcript': transcript,
            'summary': summary,
            'sentiment': sentiment,
            'status': 'success',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'groq_configured': not USE_MOCK_MODE
    })

if __name__ == '__main__':
    print("Mock Mode:", USE_MOCK_MODE)
    app.run(debug=True, host='0.0.0.0', port=5000)
