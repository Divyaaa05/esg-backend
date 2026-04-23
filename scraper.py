import requests
from bs4 import BeautifulSoup
import os
from datetime import datetime, timedelta

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def scrape_esg_news(company_name):
    articles = []
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": f"{company_name} ESG OR sustainability OR controversy",
            "from": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
            "sortBy": "relevancy",
            "language": "en",
            "pageSize": 10,
            "apiKey": NEWS_API_KEY
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if data.get("status") == "ok":
            for article in data.get("articles", []):
                articles.append({
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "source": article.get("source", {}).get("name", ""),
                    "publishedAt": article.get("publishedAt", ""),
                    "url": article.get("url", "")
                })
    except Exception as e:
        print(f"NewsAPI error: {e}")

    if len(articles) < 3:
        try:
            rss_url = f"https://news.google.com/rss/search?q={company_name}+ESG+sustainability&hl=en-IN&gl=IN&ceid=IN:en"
            response = requests.get(rss_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(response.content, "xml")
            items = soup.find_all("item")[:10]
            for item in items:
                articles.append({
                    "title": item.find("title").text if item.find("title") else "",
                    "description": item.find("description").text if item.find("description") else "",
                    "source": "Google News",
                    "publishedAt": item.find("pubDate").text if item.find("pubDate") else "",
                    "url": item.find("link").text if item.find("link") else ""
                })
        except Exception as e:
            print(f"Google News RSS error: {e}")

    return articles


def scrape_esg_reports(company_name):
    return {"has_report": False, "report_url": "", "key_stats": []}


def analyze_sentiment(articles):
    negative_keywords = [
        "controversy", "scandal", "violation", "lawsuit", "fraud", "penalty",
        "fine", "pollution", "spill", "accident", "protest", "strike",
        "greenwashing", "misleading", "unsafe", "illegal", "corruption",
        "bribery", "discrimination", "harassment", "layoff", "bankruptcy"
    ]
    positive_keywords = [
        "sustainable", "renewable", "carbon neutral", "net zero", "green",
        "award", "certification", "commitment", "initiative", "reduction",
        "clean energy", "solar", "biodiversity", "ethical"
    ]
    negative_count = 0
    positive_count = 0
    controversy_articles = []

    for article in articles:
        text = (article.get("title", "") + " " + article.get("description", "")).lower()
        neg = sum(1 for k in negative_keywords if k in text)
        pos = sum(1 for k in positive_keywords if k in text)
        negative_count += neg
        positive_count += pos
        if neg > 0:
            controversy_articles.append(article.get("title", ""))

    total = negative_count + positive_count
    if total == 0:
        controversy_score = 65
    else:
        positive_ratio = positive_count / total
        controversy_score = int(30 + (positive_ratio * 60))

    controversy_level = ("low" if controversy_score >= 65 else
                         "medium" if controversy_score >= 40 else "high")

    return {
        "realtime_controversy": min(95, max(10, controversy_score)),
        "controversy_level": controversy_level,
        "articles_found": len(articles),
        "negative_signals": negative_count,
        "positive_signals": positive_count,
        "top_controversies": controversy_articles[:3]
    }