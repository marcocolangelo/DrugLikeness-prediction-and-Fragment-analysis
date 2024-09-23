import requests
from bs4 import BeautifulSoup
import certifi
import re
import concurrent.futures
from threading import Lock

# Base URL of the webpage to scrape
base_url = "https://zinc20.docking.org/substances/subsets/in-vivo/?page="

# Set to store ZINC IDs
zinc_ids = set()
lock = Lock()

def is_valid_page(page):
    url = base_url + str(page)
    response = requests.get(url, verify=certifi.where())
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        all_links = soup.find_all('a', href=True, title=True)
        for link in all_links:
            if re.search(r'ZINC\d+', link.text):
                return True
    return False

def find_last_valid_page():
    start = 1
    end = 100000  # Set a high number to start with
    while start < end:
        mid = (start + end + 1) // 2
        if is_valid_page(mid):
            start = mid  # Move start to mid
        else:
            end = mid - 1  # Move end to mid - 1
    return start



# Find the last valid page
last_valid_page = find_last_valid_page()
print(f"The last valid page is: {last_valid_page}")



