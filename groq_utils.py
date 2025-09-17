import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("❌ GROQ_API_KEY not set in .env file")

client = Groq(api_key=GROQ_API_KEY)

def analyze_with_groq(transcript):
    try:
        # Summarization
        summary_response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an AI that summarizes customer call transcripts."},
                {"role": "user", "content": f"Summarize this transcript in 3-4 sentences:\n\n{transcript}"}
            ],
            model="llama3-8b-8192"
        )
        summary = summary_response.choices[0].message.get("content", "").strip()

        # Sentiment
        sentiment_response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Analyze the sentiment: Positive, Negative, or Neutral."},
                {"role": "user", "content": f"What's the sentiment of this transcript?\n\n{transcript}"}
            ],
            model="llama-3.1-8b-instant"
        )
        sentiment = sentiment_response.choices[0].message.get("content", "").strip()

        for word in ['Positive', 'Negative', 'Neutral']:
            if word.lower() in sentiment.lower():
                sentiment = word
                break

        return summary, sentiment

    except Exception as e:
        return f"Error: {str(e)}", "Error"
