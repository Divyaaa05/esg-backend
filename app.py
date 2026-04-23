from flask import Flask, jsonify, request
from flask_cors import CORS
from scraper import scrape_esg_news, scrape_esg_reports, analyze_sentiment
from rag_engine import generate_esg_analysis, initialize_rag
import os
import threading

app = Flask(__name__)
CORS(app)

def init_rag_background():
    try:
        initialize_rag()
        print("RAG ready.")
    except Exception as e:
        print(f"RAG init warning: {e}")

threading.Thread(target=init_rag_background, daemon=True).start()

@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json() or {}
    company = data.get("company", "").strip()
    if not company:
        return jsonify({"error": "Company name required"}), 400
    try:
        news_articles = scrape_esg_news(company)
        esg_reports = scrape_esg_reports(company)
        sentiment = analyze_sentiment(news_articles)
        result = generate_esg_analysis(company, news_articles, sentiment)
        result["news_articles"] = news_articles[:3]
        result["scraping_status"] = {
            "articles_scraped": len(news_articles),
            "has_esg_report": esg_reports.get("has_report", False),
            "live": True
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "rag": "active"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
