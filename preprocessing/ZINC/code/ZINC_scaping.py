import requests
from bs4 import BeautifulSoup
import certifi
import re

import concurrent.futures

# URL of the webpage to scrape
base_url = "https://zinc20.docking.org/substances/subsets/in-vivo/?page="

# Send GET requests to all the pages with SSL verification using certifi
zinc_ids = []

def scrape_page(page):
    url = base_url + str(page)
    response = requests.get(url, verify=certifi.where())
    print(f"Scraping page {page}...")
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        all = soup.find_all('a', href=True, title=True)
        # Find all the elements that match the pattern of the ZINC ID links
        matches = []
        for link in all:
            match = re.search(r'ZINC\d+', link.text)
            matches.append(match)
            if match:
                zinc_ids.append(match.group())

        if len(matches) == 0:
                print(f"Failed to retrieve ZINC IDs from page {page}.")
        
    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")

# Use multithreading to scrape multiple pages concurrently
with concurrent.futures.ThreadPoolExecutor() as executor:
    # Scrape pages 1 to 306
    executor.map(scrape_page, range(1, 1297))

# Print the extracted ZINC IDs
print(f"Extracted {len(zinc_ids)} ZINC IDs.")

# Save the ZINC IDs to a CSV file
with open('in_vivo_extracted_zinc_ids_new.csv', 'w') as output_file:
    for zinc_id in zinc_ids:
        output_file.write(zinc_id + '\n')
