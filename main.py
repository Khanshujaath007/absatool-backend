from pyabsa import (
    available_checkpoints,
    ATEPCCheckpointManager,
)

import requests
from bs4 import BeautifulSoup
import logging
from urllib.parse import urlparse

def get_reviews(URL: str, total_reviews_needed: int) -> list:
    headers = {
        "authority": "www.amazon.com",
        "pragma": "no-cache",
        "cache-control": "no-cache",
        "dnt": "1",
        "upgrade-insecure-requests": "1",
        "user-agent": "Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36",
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "sec-fetch-site": "none",
        "sec-fetch-mode": "navigate",
        "sec-fetch-dest": "document",
        "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
    }

    def get_page_html(page_url: str) -> str:
        resp = requests.get(page_url, headers=headers)
        return resp.text

    html = get_page_html(URL)
    soup = BeautifulSoup(html, 'html.parser')

    a_tag = soup.find('a', {'data-hook': 'see-all-reviews-link-foot'})
    if a_tag:
        BASE_URL = "https://www.amazon.in" + a_tag.get('href')
        print(BASE_URL)
    else:
        print("No matching <a> tag found.")
        return []

    def get_reviews_from_html(page_html: str) -> list:
        soup = BeautifulSoup(page_html, "lxml")
        reviews = soup.find_all("div", {"class": "a-section celwidget"})
        return reviews

    def get_review_text(soup_object: BeautifulSoup) -> str:
        review_text = soup_object.find(
            "span", {"class": "a-size-base review-text review-text-content"}
        ).get_text()
        return review_text.strip()

    all_results = []
    page_no = 0

    page_url = get_base_url(BASE_URL, page_no)
    print(page_url)
    all_star_links = get_star_links(get_page_html(page_url))

    for star_rating, href_link in all_star_links.items():
        star_reviews_collected = 0

        while star_reviews_collected < 30 and len(all_results) < total_reviews_needed:
            star_url = "https://www.amazon.in" + href_link
            logging.info(f"Scraping page {star_url}")

            html = get_page_html(star_url)
            reviews = get_reviews_from_html(html)

            for rev in reviews:
                review_text = get_review_text(rev)
                all_results.append(review_text)
                star_reviews_collected += 1
                if len(all_results) >= total_reviews_needed:
                    break

            page_no += 1
            next_page_link = get_base_url(BASE_URL, page_no)
            if not next_page_link:
                print("No more reviews found.")
                break
            page_url = next_page_link
            print(next_page_link)

    return all_results

def get_base_url(BASE_URL,page_no) -> str:
    parsed_url = urlparse(BASE_URL)
    base_url = parsed_url.scheme + "://" + parsed_url.netloc + parsed_url.path
    new_url = f"{base_url}?pageNumber={page_no}&reviewerType=all_reviews"
    return new_url

def get_star_links(html) -> dict:
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table', {'id': 'histogramTable'})
    star_ratings_links = {}

    rows = table.find_all('tr')

    for row in rows:
        link = row.find('a', class_='a-link-normal')
        if link:
            star_rating = link.text.strip()
            href_link = link['href']
            star_rating_int = int(star_rating.split()[0])
            star_ratings_links[star_rating_int] = href_link

    return star_ratings_links

# URL = "https://www.amazon.in/iQOO-Storage-Snapdragon-Processor-44WFlashCharge/dp/B07WHS7MZ4/ref=cm_cr_arp_d_product_top?ie=UTF8&th=1"
# total_reviews_needed = 20
# reviews = get_reviews(URL, total_reviews_needed)
# print(reviews)


"""
MLModel takes list of strings as parameter and performs the proceesing on data, and
return a object with aspect terms, sentiment of these aspects ans confidence of these terms 
processed by the model
"""


def aggregatePayload(result):
    payload = {
        "Aspect Terms": [],
        "Sentiment of Aspects": [],
        "Confidence of Aspect": [],
    }
    for i in range(len(result)):
        for j in range(len(result[i]["aspect"])):
            payload["Aspect Terms"].append(result[i]["aspect"][j])
            payload["Sentiment of Aspects"].append(result[i]["sentiment"][j])
            payload["Confidence of Aspect"].append(result[i]["confidence"][j])

    return payload


def MLModel(data):
    check_point_map = available_checkpoints()
    if data != []:
        aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(
            checkpoint="english", auto_device=True
        )
        atepc_result = aspect_extractor.batch_predict(
            target_file=data, pred_sentiment=True, print_result=False, save_result=False
        )
        # iterate through all the atepc result objects and build payload
        finalPayload = aggregatePayload(atepc_result)
        return finalPayload
    else:
        return {"response": "Data not found"}

__all__ = ["MLModel"]
