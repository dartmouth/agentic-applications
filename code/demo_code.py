import requests
from bs4 import BeautifulSoup


def get_headlines(url):
    """
    Retrieves headlines from a website by targeting the 'headline' CSS class.

    Returns:
        list: A list of headline strings
    """

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Target elements with the 'headline' class specifically
        headline_elements = soup.select(".headline")

        headlines = [headline.get_text().strip() for headline in headline_elements]

        return headlines

    except Exception as e:
        print(f"Error retrieving headlines: {e}")
        return []


# Example usage
if __name__ == "__main__":
    headlines = get_headlines(url="https://www.thedartmouth.com")
    print(f"Found {len(headlines)} headlines:")
    for i, headline in enumerate(headlines, 1):
        print(f"{i}. {headline}")
