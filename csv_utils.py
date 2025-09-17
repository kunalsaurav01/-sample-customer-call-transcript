import csv
import os

def save_to_csv(transcript, summary, sentiment, filename="call_analysis.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Transcript", "Summary", "Sentiment"])
        writer.writerow([transcript, summary, sentiment])
