from __future__ import annotations

from typing import Any


GDELT_COMPANY_QUERY_TERMS = {
    "A": ["Agilent Technologies", "Agilent"],
    "AA": ["Alcoa"],
    "AAPL": ["Apple", "Apple Inc"],
    "ABBV": ["AbbVie"],
    "ABT": ["Abbott Laboratories", "Abbott"],
    "ACN": ["Accenture"],
    "ADSK": ["Autodesk"],
    "AEP": ["American Electric Power"],
    "AFL": ["Aflac"],
    "AKAM": ["Akamai Technologies", "Akamai"],
    "ALB": ["Albemarle"],
    "ALL": ["Allstate"],
    "AMAT": ["Applied Materials"],
    "AMD": ["Advanced Micro Devices", "AMD"],
    "AMT": ["American Tower"],
    "AMZN": ["Amazon"],
    "BA": ["Boeing"],
    "BAC": ["Bank of America"],
    "BBY": ["Best Buy"],
    "BMY": ["Bristol Myers Squibb"],
    "BP": ["BP"],
    "BRK-A": ["Berkshire Hathaway"],
    "BRK-B": ["Berkshire Hathaway"],
    "CAT": ["Caterpillar"],
    "CRM": ["Salesforce"],
    "CSCO": ["Cisco Systems", "Cisco"],
    "CVX": ["Chevron"],
    "D": ["Dominion Energy"],
    "F": ["Ford Motor", "Ford"],
    "GLD": ["SPDR Gold Shares"],
    "GOOGL": ["Alphabet", "Google"],
    "HD": ["Home Depot"],
    "JNJ": ["Johnson & Johnson"],
    "JPM": ["JPMorgan Chase"],
    "KO": ["Coca-Cola"],
    "MA": ["Mastercard"],
    "META": ["Meta Platforms", "Facebook"],
    "MRK": ["Merck"],
    "MSFT": ["Microsoft"],
    "NFLX": ["Netflix"],
    "NVDA": ["Nvidia"],
    "ORCL": ["Oracle"],
    "PEP": ["PepsiCo"],
    "PG": ["Procter & Gamble"],
    "SPY": ["SPDR S&P 500 ETF"],
    "TLT": ["iShares 20+ Year Treasury Bond ETF"],
    "TSLA": ["Tesla"],
    "UNH": ["UnitedHealth Group"],
    "V": ["Visa"],
    "VZ": ["Verizon"],
    "WMT": ["Walmart"],
    "XLE": ["Energy Select Sector SPDR Fund"],
    "XLB": ["Materials Select Sector SPDR Fund"],
    "XLP": ["Consumer Staples Select Sector SPDR Fund"],
    "XLU": ["Utilities Select Sector SPDR Fund"],
    "XLY": ["Consumer Discretionary Select Sector SPDR Fund"],
    "XOM": ["Exxon Mobil"],
}

GDELT_AMBIGUOUS_SYMBOLS = {
    "A", "AA", "ALL", "AN", "ARE", "AT", "BALL", "CAT", "D", "F",
    "GLD", "GOLD", "GPS", "HE", "IT", "KEY", "L", "LOW", "NOW",
    "ON", "SEE", "SPY", "T", "TD", "TEAM", "TER", "V", "YOU",
}

def gdelt_query_terms(symbol: str) -> list[str]:
    normalized = symbol.strip().upper()
    if normalized in GDELT_COMPANY_QUERY_TERMS:
        return GDELT_COMPANY_QUERY_TERMS[normalized]
    if normalized in GDELT_AMBIGUOUS_SYMBOLS or len(normalized) <= 3:
        return []
    return [normalized]

def _gdelt_query_text(terms: list[str]) -> str:
    return " OR ".join(f'"{term}"' if " " in term else term for term in terms)

def _alpha_vantage_news_time(value: str, *, end_of_day: bool) -> str:
    text = str(value or "").strip()
    compact = text.replace("-", "").replace(":", "")
    if "T" in compact and len(compact) >= 13:
        return compact[:13]
    date_part = text[:10].replace("-", "")
    suffix = "T2359" if end_of_day else "T0000"
    return f"{date_part}{suffix}"
