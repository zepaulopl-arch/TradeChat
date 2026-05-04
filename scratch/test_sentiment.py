import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
import feedparser

def test():
    nltk.download("vader_lexicon", quiet=True)
    sid = SentimentIntensityAnalyzer()
    translator = GoogleTranslator(source="pt", target="en")
    
    ticker = "PETR4"
    url = f"https://news.google.com/rss/search?q={ticker}+Bovespa&hl=pt-BR&gl=BR&ceid=BR:pt-419"
    feed = feedparser.parse(url)
    
    print(f"Encontradas {len(feed.entries)} notícias para {ticker}")
    for entry in feed.entries[:5]:
        title = entry.title
        try:
            translated = translator.translate(title)
            score = sid.polarity_scores(translated)["compound"]
            print(f"Original: {title[:50]}...")
            print(f"Traduzido: {translated[:50]}...")
            print(f"Score: {score}")
            print("-" * 30)
        except Exception as e:
            print(f"Erro: {e}")

if __name__ == "__main__":
    test()
