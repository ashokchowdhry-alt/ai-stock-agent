import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import requests

# -------------------------
# CONFIG
# -------------------------

STOCKS = [
    "AAPL","MSFT","GOOGL","AMZN","TSLA",
    "NVDA","META","AMD","NFLX","INTC",
    "JPM","V","MA","DIS","BA",
    "NKE","UBER","CRM","ORCL","PYPL"
]

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
TO_EMAIL = os.getenv("TO_EMAIL")

# -------------------------
# SCORING FUNCTION
# -------------------------

def calculate_score(df):
    df = df.copy()

    # Ensure proper column selection
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["return_30"] = df["Close"].pct_change(30)
    df["return_7"] = df["Close"].pct_change(7)
    df["volatility"] = df["Close"].pct_change().rolling(30).std()
    df["volume_trend"] = df["Volume"].pct_change(7)

    df = df.dropna()

    if df.empty:
        return None

    latest = df.iloc[-1]

    score = (
        0.4 * float(latest["return_30"]) +
        0.2 * float(latest["return_7"]) -
        0.2 * float(latest["volatility"]) +
        0.2 * float(latest["volume_trend"])
    )

    return round(score * 100, 2)

# -------------------------
# FETCH DATA + RANK
# -------------------------

def analyze_stocks():
    results = []

    for ticker in STOCKS:
        try:
            df = yf.download(ticker, period="3mo", auto_adjust=True, progress=False)

    ##     df = yf.download(ticker, period="3mo", progress=False)

            if df.empty or len(df) < 40:
                continue

            df = df.dropna()

            if len(df) < 40:
                continue

            score = calculate_score(df)

            if pd.isna(score):
                continue

            results.append({"ticker": ticker, "score": float(score)})

        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

    if not results:
        raise Exception("No stock data processed successfully.")

    df_scores = pd.DataFrame(results)
    df_scores = df_scores.sort_values(by="score", ascending=False)

    return df_scores


# -------------------------
# LLM SUMMARY (Groq)
# -------------------------

def generate_summary(df_scores):

    top5 = df_scores.head(5).to_string(index=False)
    bottom3 = df_scores.tail(3).to_string(index=False)

    prompt = f"""
You are a financial AI analyst.

Top 5 stocks:
{top5}

Bottom 3 stocks:
{bottom3}

Provide:
1. Overall portfolio tone
2. Risk observation
3. Short recommendation
"""

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}]
        }
    )

    data = response.json()

    # DEBUG PRINT
    print("Groq response:", data)

    if "choices" not in data:
        raise Exception(f"Groq API Error: {data}")

    return data["choices"][0]["message"]["content"]


def generate_summary2(df_scores):

    top5 = df_scores.head(5).to_string(index=False)
    bottom3 = df_scores.tail(3).to_string(index=False)

    prompt = f"""
You are a financial AI analyst.

Top 5 stocks:
{top5}

Bottom 3 stocks:
{bottom3}

Provide:
1. Overall portfolio tone
2. Risk observation
3. Short recommendation
"""

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}]
        }
    )

    return response.json()["choices"][0]["message"]["content"]

# -------------------------
# SEND EMAIL
# -------------------------

def send_email(summary, df_scores):

    msg = MIMEMultipart()      
    msg["From"] = EMAIL_USER
    msg["To"] = TO_EMAIL
    msg["Subject"] = "Daily AI Stock Ranking"

    html = f"""
    <h2>Stock Ranking</h2>
    {df_scores.to_html(index=False)}
    <hr>
    <h3>AI Summary</h3>
    <p>{summary}</p>
    """

    msg.attach(MIMEText(html, "html"))

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(EMAIL_USER, EMAIL_PASS)
    server.sendmail(EMAIL_USER, TO_EMAIL, msg.as_string())
    server.quit()

# -------------------------
# MAIN
# -------------------------

if __name__ == "__main__":
    df_scores = analyze_stocks()
    summary = generate_summary(df_scores)
    send_email(summary, df_scores)
