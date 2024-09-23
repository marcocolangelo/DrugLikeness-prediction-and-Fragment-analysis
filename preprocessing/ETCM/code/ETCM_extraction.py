from bs4 import BeautifulSoup
import re
import tqdm
import requests
import csv

# Carica il file HTML per testare la funzione
url = "http://www.tcmip.cn/ETCM/index.php/Home/Index/cf_details.html?id=1"

def scraper(html_content):
    # Crea un oggetto BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Trova lo script che contiene l'oggetto data
    script_tag = soup.find('script', string=re.compile(r"data\s*:\s*\["))
    data_text = script_tag.string

    # Estrai il contenuto dell'oggetto data
    data_match = re.search(r"data\s*:\s*(\[.*?\])", data_text, re.DOTALL)
    data_json = data_match.group(1)

    # Trova tutti i dizionari nell'oggetto data usando regex
    pattern = re.compile(r'\{"ID":".*?","Item Name":".*?"\}')
    matches = pattern.findall(data_json)

    # Crea una lista di dizionari dal testo trovato
    data_list = []
    for match in matches:
        item_id = re.search(r'"ID":"(.*?)"', match).group(1)
        item_name = re.search(r'"Item Name":"(.*?)"', match).group(1)
        # Decodifica le entità HTML
        item_name = BeautifulSoup(item_name, 'html.parser').text
        data_list.append({"ID": item_id, "Item Name": item_name})

    return data_list

# Funzione per estrarre il valore corrispondente a un dato ID
def extract_value(data_list, id_value):
    for item in data_list:
        if f"<div id='{id_value}'" in item["ID"]:
            return item["Item Name"].strip()
    return None

# Estrai le informazioni richieste
def extract_ingredient_info(html_content):
    data_list = scraper(html_content)

    ingredient_info = {
        "Ingredient Name in English": extract_value(data_list, 'L771'),
        "Molecular Formula": extract_value(data_list, 'L773'),
        "Diseases Associated with This Ingredient": extract_value(data_list, 'L7727'),
        "External Link to PubChem": extract_value(data_list, 'L7729'),
        "Formulas Containing This Ingredient": extract_value(data_list, 'L7733')
    }
    
    diseases_entry = next((item for item in data_list if "<div id='L7727'" in item["ID"]), None)
    if diseases_entry:
        item_name = diseases_entry["Item Name"]
        link_tags = []
        for name in item_name.split(','):
            link_tags.append(name)
        #togli NA da link_tags
        link_tags = [x for x in link_tags if x != 'NA']
        ingredient_info["Diseases Associated with This Ingredient"] = len(link_tags)

    pubchem_entry = next((item for item in data_list if "<div id='L7729'" in item["ID"]), None)
    if pubchem_entry:
        link_tag = BeautifulSoup(pubchem_entry["Item Name"], 'html.parser').find('a')
        if link_tag:
            ingredient_info["External Link to PubChem"] = link_tag['href']

    formulas_entry = next((item for item in data_list if "<div id='L7733'" in item["ID"]), None)
    if formulas_entry:
        item_name = formulas_entry["Item Name"]
        link_tags = []
        for name in item_name.split(','):
            link_tags.append(name)

        link_tags = [x for x in link_tags if x != 'NA']
        #elimina '' dalla lista
        link_tags = [x for x in link_tags if x != '']
        ingredient_info["Formulas Containing This Ingredient"] = len(link_tags)

    return ingredient_info

# Estrai le informazioni dal file HTML di test
data = []
for comp_id in tqdm.tqdm(range(1, 7285)):
    url = f"http://www.tcmip.cn/ETCM/index.php/Home/Index/cf_details.html?id={comp_id}"
    response = requests.get(url)
    html_content = response.content.decode('utf-8')
    ingredient_info = extract_ingredient_info(html_content)
    data.append(ingredient_info)

# Scrivi i dati in un file CSV
filename = "ETCM_data.csv"
fields = ["Ingredient Name in English", "Molecular Formula", "Diseases Associated with This Ingredient", "External Link to PubChem", "Formulas Containing This Ingredient"]

with open(filename, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()
    writer.writerows(data)





