import requests
from bs4 import BeautifulSoup


def analyze(url: str):
    try:
        # Fetch webpage
        response = requests.get(url)

        if response.status_code != 200:
            return {
                "success": False,
                "message": "Could not fetch article"
            }

        # Parse HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # Get title
        title = soup.title.string if soup.title else "No title found"

        # VERY BASIC demo logic
        suspicious_words = [
            "shocking",
            "miracle",
            "secret",
            "100%",
            "breaking"
        ]

        title_lower = title.lower()

        fake_score = 0

        for word in suspicious_words:
            if word in title_lower:
                fake_score += 20

        fake = fake_score >= 40

        return {
            "success": True,
            "title": title,
            "fake": fake,
            "confidence": fake_score
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }