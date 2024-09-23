import requests
from bs4 import BeautifulSoup
import certifi
import csv
import concurrent.futures
from tqdm import tqdm
from threading import Lock
import time
import logging

# Configura il logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("terminal_output.txt"),
    logging.StreamHandler()
])

# File di input e output
input_file = 'ZINC\\new\\ZINC_in_vivo_nuovi_samples_labelled.csv'  # Sostituisci con il percorso del tuo file di input
output_file = 'ZINC\\new\\ZINC_in_vivo_nuovi_samples_smiles_output.csv'

# Base URL per accedere alle pagine delle sostanze
base_url = "https://zinc15.docking.org/substances/"

# Lista per memorizzare i risultati
results = []
lock = Lock()

def get_smiles(zinc_id, label):
    url = base_url + zinc_id + "/"
    retries = 5
    for attempt in range(retries):
        try:
            response = requests.get(url, verify=certifi.where(), timeout=10)  # Imposta un timeout di 10 secondi
            response.raise_for_status()  # Verifica che la richiesta sia stata eseguita con successo
            soup = BeautifulSoup(response.content, 'html.parser')
            smiles_field = soup.find('input', {'id': 'substance-smiles-field'})
            if smiles_field and 'value' in smiles_field.attrs:
                if attempt > 0:
                    logging.info(f"Successfully retrieved SMILES for {zinc_id} on attempt {attempt + 1}")
                return (zinc_id, smiles_field['value'], label)
            else:
                logging.warning(f"SMILES not found for {zinc_id}")
                return (zinc_id, None, label)
        except requests.RequestException as e:
            logging.warning(f"Attempt {attempt + 1} failed for {zinc_id}. Error: {e}")
            time.sleep(1)  # Aspetta 1 secondo prima di riprovare
    logging.error(f"Failed to retrieve page for {zinc_id} after {retries} attempts.")
    return (zinc_id, None, label)

# Funzione per gestire i risultati in modo thread-safe
def handle_result(future):
    result = future.result()
    with lock:
        results.append(result)

# Leggi il file CSV di input e crea una lista di ID ZINC e label
zinc_data = []
with open(input_file, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)  # Skip header
    for row in csvreader:
        zinc_data.append((row[0], row[1]))  # Aggiungi tuple (ZINCID, label)

# Usa il multithreading per eseguire lo scraping
num_threads = 50
with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(get_smiles, zinc_id, label) for zinc_id, label in zinc_data]
    # Usa tqdm per mostrare la barra di avanzamento
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Progress"):
        handle_result(future)

# Scrivi i risultati in un nuovo file CSV
with open(output_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['ZINCID', 'Smiles', 'Drug'])  # Header
    csvwriter.writerows([result for result in results if result[1] is not None])

logging.info("Scraping complete. Results saved to %s", output_file)
