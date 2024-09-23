import hashlib
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem import DataStructs
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from rdkit.DataStructs import ExplicitBitVect
from rdkit.Chem import Draw
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from collections import Counter
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm  # Importa tqdm per la barra di avanzamento

# Funzione per convertire numpy array in ExplicitBitVect
def numpy_to_bitvect(np_array):
    bitvect = ExplicitBitVect(len(np_array))
    for i, bit in enumerate(np_array):
        if bit == 1:
            bitvect.SetBit(i)
    return bitvect

# Funzione per concatenare le fingerprints
def concatenate_fingerprints(ecfp4, fcfp4, wc_bitvect=None):
    if wc_bitvect is None:
        combined_fp = ExplicitBitVect(len(ecfp4) + len(fcfp4))
    else:
        combined_fp = ExplicitBitVect(len(ecfp4) + len(fcfp4) + len(wc_bitvect))
    
    # Aggiungi ECFP4
    for i in range(len(ecfp4)):
        if ecfp4.GetBit(i):
            combined_fp.SetBit(i)

    # Aggiungi wc_bitvect
    if wc_bitvect is not None:
        offset = len(ecfp4)
        for i in range(len(wc_bitvect)):
            if wc_bitvect.GetBit(i):
                combined_fp.SetBit(offset + i)

    # Aggiungi FCFP4
    if wc_bitvect is not None:
        offset += len(wc_bitvect)
    else:
        offset = len(ecfp4)
    for i in range(len(fcfp4)):
        if fcfp4.GetBit(i):
            combined_fp.SetBit(offset + i)

    return combined_fp

# Funzione per codificare il numero di wildcard
def wildcard_to_n_bit(wildcard_count, n_bits=64):
    wildcard_str = str(wildcard_count)
    hash_object = hashlib.sha256(wildcard_str.encode())
    binary_hash = bin(int(hash_object.hexdigest(), 16))[2:].zfill(256)
    repeated_hash = (binary_hash * (n_bits // len(binary_hash) + 1))[:n_bits]
    fingerprint_array = np.array([int(bit) for bit in repeated_hash])
    return fingerprint_array

# Funzione per calcolare extended fingerprint usando le funzioni standard di RDKit
def compute_extended_fp(smiles, num_neigh, n_bits=1024, use_features=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Errore: Impossibile analizzare SMILES: {smiles}")
        return None
    
    # Genera ECFP4 utilizzando AllChem.GetMorganFingerprintAsBitVect
    ecfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits, useFeatures=use_features)
    
    # Genera FCFP4 utilizzando AllChem.GetMorganFingerprintAsBitVect con useFeatures=True
    fcfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits, useFeatures=True)
    
    # Converti i fingerprint in numpy array
    ecfp4_numpy = np.array([int(bit) for bit in ecfp4])
    ecfp4_numpy_weighted = np.tile(ecfp4_numpy, 2)
    ecfp4_weighted = numpy_to_bitvect(ecfp4_numpy_weighted)
    
    # Concatenazione dei fingerprint
    concatenated_fingerprint = concatenate_fingerprints(ecfp4_weighted, fcfp4)
    
    return concatenated_fingerprint

# Funzione per calcolare la media della similarità di Tanimoto
def calculate_average_tanimoto_similarity(tanimoto_matrix, exclude_diagonal=True):
    if exclude_diagonal:
        upper_triangle = tanimoto_matrix[np.triu_indices(tanimoto_matrix.shape[0], k=1)]
        mean_similarity = np.mean(upper_triangle)
    else:
        mean_similarity = np.mean(tanimoto_matrix)
    
    return mean_similarity

# Funzione per calcolare la matrice di similarità di Tanimoto
def tanimoto_similarity_matrix(fingerprints):
    n = len(fingerprints)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim  # Simmetrico
    
    return similarity_matrix

# Funzione per eseguire il clustering basato su Tanimoto
def perform_tanimoto_clustering(fingerprints, dataset, show=False):
    print(f"Inizio clustering Tanimoto per il dataset: {dataset}")
    labels, cluster_counts, tanimoto_matrix = dbscan_clustering(fingerprints, eps=0.6, min_samples=2)

    mean_similarity = calculate_average_tanimoto_similarity(tanimoto_matrix)
    print(f"Similarità media di Tanimoto per il dataset {dataset}: {mean_similarity}")

    if show:
        plt.imshow(tanimoto_matrix, cmap='hot', interpolation='nearest')
        plt.title(f"Matrice di Similarità di Tanimoto per il dataset {dataset}")
        plt.colorbar()
        plt.savefig(f"data/fragment_analysis/tanimoto_similarity_matrix_{dataset}.png")
        plt.show()

    print(f"Clustering Tanimoto completato per il dataset: {dataset}")
    return labels, cluster_counts

# Funzione per eseguire il clustering DBSCAN
def dbscan_clustering(fingerprints, eps=0.3, min_samples=2):
    print(f"Inizio clustering DBSCAN con eps={eps} e min_samples={min_samples}")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    # Usa la matrice di distanza Tanimoto per il clustering
    tanimoto_matrix = tanimoto_similarity_matrix(fingerprints)
    distance_matrix = 1 - tanimoto_matrix
    labels = dbscan.fit_predict(distance_matrix)

    # Calcola l'indice di silhouette (facoltativo, solo se ci sono più cluster)
    if len(set(labels)) > 1:
        silhouette_avg = silhouette_score(distance_matrix, labels, metric="precomputed")
        print(f"Silhouette Score: {silhouette_avg}")

    unique, counts = np.unique(labels, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    print(f"Conteggio dei cluster: {cluster_counts}")
    print("Clustering DBSCAN completato")
    return labels, cluster_counts, tanimoto_matrix

# Funzione per salvare le immagini dei frammenti
def save_smiles_images(smiles_list, labels, image_folder):
    print(f"Salvataggio delle immagini dei frammenti in: {image_folder}")
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
        print(f"Creata la cartella: {image_folder}")

    else:
        #elimina i file esistenti
        for file in os.listdir(image_folder):
            os.remove(os.path.join(image_folder, file))
            
    
    image_paths = []
    for i, (smiles, cluster) in enumerate(zip(smiles_list, labels)):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Errore: Impossibile generare molecola da SMILES: {smiles}")
            continue
        
        img = Draw.MolToImage(mol, size=(300, 300))
        img_path = os.path.join(image_folder, f"fragment_{i}_cluster_{cluster}.png")
        img.save(img_path)
        image_paths.append(img_path)
    
    print(f"Salvato {len(image_paths)} immagini dei frammenti in {image_folder}")
    return image_paths

# Funzione per calcolare la fingerprint per una singola riga
def compute_fp_row(row, n_bits=1024):
    smiles = row["smiles"]
    num_neigh = row["num_neigh"]
    pred = row["pred"]
    mol_id = row["id_molecola"]  # Ottieni l'ID della molecola
    real_label = row.get("real_label", None)  # Ottieni real_label se disponibile
    fp = compute_extended_fp(smiles, num_neigh, n_bits=n_bits)
    if fp is None:
        return None  # Indica un errore
    return {
        'pred': pred,
        'fingerprint': fp,
        'smiles': smiles,
        'id': mol_id,
        'real_label': real_label
    }

# Funzione per estrarre le fingerprint con deduplicazione e barra di avanzamento
def extract_fp(df, n_bits=1024, max_workers=4):
    print("Inizio estrazione e deduplicazione delle fingerprint")
    fp_drug_dict = {}
    fp_non_drug_dict = {}
    errors = 0

    # Usa ProcessPoolExecutor per la generazione parallela delle fingerprint
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Funzione parziale per includere n_bits
        compute_func = partial(compute_fp_row, n_bits=n_bits)
        # Invia tutti i task
        futures = {executor.submit(compute_func, row): idx for idx, row in df.iterrows()}
        
        # Inizializza la barra di avanzamento con tqdm
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generazione delle Fingerprint"):
            result = future.result()
            if result is None:
                errors += 1
                continue
            pred = result['pred']
            fp = result['fingerprint']
            smiles = result['smiles']
            mol_id = result['id']
            real_label = result['real_label']
            
            # Converti la fingerprint in una rappresentazione stringa unica
            fp_str = DataStructs.BitVectToText(fp)
            
            if pred == 1:
                if fp_str not in fp_drug_dict:
                    fp_drug_dict[fp_str] = {
                        'fingerprint': fp,
                        'smiles_list': [],
                        'id_list': [],
                        'real_label_list': [],
                        'count': 0
                    }
                fp_drug_dict[fp_str]['smiles_list'].append(smiles)
                fp_drug_dict[fp_str]['id_list'].append(mol_id)
                fp_drug_dict[fp_str]['real_label_list'].append(real_label)
                fp_drug_dict[fp_str]['count'] += 1
            else:
                if fp_str not in fp_non_drug_dict:
                    fp_non_drug_dict[fp_str] = {
                        'fingerprint': fp,
                        'smiles_list': [],
                        'id_list': [],
                        'real_label_list': [],
                        'count': 0
                    }
                fp_non_drug_dict[fp_str]['smiles_list'].append(smiles)
                fp_non_drug_dict[fp_str]['id_list'].append(mol_id)
                fp_non_drug_dict[fp_str]['real_label_list'].append(real_label)
                fp_non_drug_dict[fp_str]['count'] += 1

    print("Estrazione e deduplicazione delle fingerprint completata")
    print(f"Fingerprint duplicate per Drug: {len(fp_drug_dict)} uniche")
    print(f"Fingerprint duplicate per Non-Drug: {len(fp_non_drug_dict)} uniche")
    print(f"Numero totale di errori durante l'estrazione: {errors}")
    return fp_drug_dict, fp_non_drug_dict, errors

# Funzione aggiornata per analizzare i frammenti con multithreading, deduplicazione, barra di avanzamento e stampe di avanzamento
def analyze_fragments(fragments_path, output_excel_drug, output_excel_non_drug, image_folder_drug, image_folder_non_drug, show=False, max_workers=4):
    print("Inizio analisi dei frammenti")
    df = pd.read_csv(fragments_path)
    print(f"Caricato il dataset con {len(df)} righe")

    fp_drug_dict, fp_non_drug_dict, errors = extract_fp(df, n_bits=1024, max_workers=max_workers)

    # Converti i dizionari in liste per il clustering
    fp_drug_unique = [v['fingerprint'] for v in fp_drug_dict.values()]
    fp_non_drug_unique = [v['fingerprint'] for v in fp_non_drug_dict.values()]

    print(f"Numero di fingerprint uniche Drug: {len(fp_drug_unique)}")
    print(f"Numero di fingerprint uniche Non-Drug: {len(fp_non_drug_unique)}")

    # Clustering per il gruppo drug
    print("Inizio clustering per il gruppo Drug")
    drug_labels, _ = perform_tanimoto_clustering(fp_drug_unique, "drug", show=show)
    print("Clustering per il gruppo Drug completato")

    # Clustering per il gruppo non-drug
    print("Inizio clustering per il gruppo Non-Drug")
    non_drug_labels, _ = perform_tanimoto_clustering(fp_non_drug_unique, "non_drug", show=show)
    print("Clustering per il gruppo Non-Drug completato")

    # Salva i Cluster Drug in Excel
    print(f"Inizio salvataggio dei risultati Drug in Excel: {output_excel_drug}")
    wb_drug = Workbook()
    ws_drug = wb_drug.active
    ws_drug.title = "Fragments Clusters"

    ws_drug.append(['SMILES', 'Cluster', 'ID Molecola', 'Real Label', 'Occurrence', 'Image'])  # Aggiungi "Occurrence"

    for i, (fp_info, cluster_label) in enumerate(zip(fp_drug_dict.values(), drug_labels)):
        # Per ogni fingerprint unica
        smiles_combined = "; ".join(fp_info['smiles_list'])
        ids_combined = "; ".join(map(str, fp_info['id_list']))
        real_labels_combined = "; ".join(map(str, fp_info['real_label_list']))
        count = fp_info['count']

        # Genera un'immagine per uno dei SMILES (rappresentativo)
        mol = Chem.MolFromSmiles(fp_info['smiles_list'][0])
        if mol is None:
            print(f"Errore: Impossibile generare molecola da SMILES: {fp_info['smiles_list'][0]}")
            img_path = ""
        else:
            img = Draw.MolToImage(mol, size=(300, 300))
            img_path = os.path.join(image_folder_drug, f"unique_fragment_{i}_cluster_{cluster_label}.png")
            if not os.path.exists(image_folder_drug):
                os.makedirs(image_folder_drug)
                print(f"Creata la cartella per le immagini Drug: {image_folder_drug}")
            img.save(img_path)
        
        ws_drug.append([smiles_combined, cluster_label, ids_combined, real_labels_combined, count])
        if img_path:
            img_obj = Image(img_path)
            img_obj.height = 100
            img_obj.width = 100
            ws_drug.add_image(img_obj, f"F{i+2}")  # Colonna F per l'immagine

    wb_drug.save(output_excel_drug)
    print(f"Salvataggio dei risultati Drug completato: {output_excel_drug}")

    # Salva i Cluster Non-Drug in Excel
    print(f"Inizio salvataggio dei risultati Non-Drug in Excel: {output_excel_non_drug}")
    wb_non_drug = Workbook()
    ws_non_drug = wb_non_drug.active
    ws_non_drug.title = "Fragments Clusters"

    ws_non_drug.append(['SMILES', 'Cluster', 'ID Molecola', 'Real Label', 'Occurrence', 'Image'])  # Aggiungi "Occurrence"

    for i, (fp_info, cluster_label) in enumerate(zip(fp_non_drug_dict.values(), non_drug_labels)):
        # Per ogni fingerprint unica
        smiles_combined = "; ".join(fp_info['smiles_list'])
        ids_combined = "; ".join(map(str, fp_info['id_list']))
        real_labels_combined = "; ".join(map(str, fp_info['real_label_list']))
        count = fp_info['count']

        # Genera un'immagine per uno dei SMILES (rappresentativo)
        mol = Chem.MolFromSmiles(fp_info['smiles_list'][0])
        if mol is None:
            print(f"Errore: Impossibile generare molecola da SMILES: {fp_info['smiles_list'][0]}")
            img_path = ""
        else:
            img = Draw.MolToImage(mol, size=(300, 300))
            img_path = os.path.join(image_folder_non_drug, f"unique_fragment_{i}_cluster_{cluster_label}.png")
            if not os.path.exists(image_folder_non_drug):
                os.makedirs(image_folder_non_drug)
                print(f"Creata la cartella per le immagini Non-Drug: {image_folder_non_drug}")
            img.save(img_path)
        
        ws_non_drug.append([smiles_combined, cluster_label, ids_combined, real_labels_combined, count])
        if img_path:
            img_obj = Image(img_path)
            img_obj.height = 100
            img_obj.width = 100
            ws_non_drug.add_image(img_obj, f"F{i+2}")  # Colonna F per l'immagine

    wb_non_drug.save(output_excel_non_drug)
    print(f"Salvataggio dei risultati Non-Drug completato: {output_excel_non_drug}")

    # Crea riepilogo cluster per il gruppo drug
    print("Creazione del riepilogo dei cluster per il gruppo Drug")
    cluster_summary_drug = pd.DataFrame({
        'Cluster': list(set(drug_labels)),
        'Num_Elements': list(Counter(drug_labels).values()),
        'Num_Unique_SMILES': [sum(1 for fp_info in fp_drug_dict.values() if cluster_label == label) 
                              for label in set(drug_labels)]
    })

    # Crea riepilogo cluster per il gruppo non-drug
    print("Creazione del riepilogo dei cluster per il gruppo Non-Drug")
    cluster_summary_non_drug = pd.DataFrame({
        'Cluster': list(set(non_drug_labels)),
        'Num_Elements': list(Counter(non_drug_labels).values()),
        'Num_Unique_SMILES': [sum(1 for fp_info in fp_non_drug_dict.values() if cluster_label == label) 
                              for label in set(non_drug_labels)]
    })

    # Aggiungi riepilogo cluster al file drug
    print("Aggiunta del riepilogo dei cluster al file Excel Drug")
    with pd.ExcelWriter(output_excel_drug, engine='openpyxl', mode='a') as writer_drug:
        cluster_summary_drug.to_excel(writer_drug, sheet_name="Cluster Summary", index=False)
    print("Riepilogo dei cluster aggiunto al file Excel Drug")

    # Aggiungi riepilogo cluster al file non-drug
    print("Aggiunta del riepilogo dei cluster al file Excel Non-Drug")
    with pd.ExcelWriter(output_excel_non_drug, engine='openpyxl', mode='a') as writer_non_drug:
        cluster_summary_non_drug.to_excel(writer_non_drug, sheet_name="Cluster Summary", index=False)
    print("Riepilogo dei cluster aggiunto al file Excel Non-Drug")

    print("Analisi dei frammenti completata")
    return errors

# Funzione per la ricerca degli iperparametri (invariata)
def dbscan_hyperparameter_search(fingerprints, eps_values, min_samples_values):
    best_eps = None
    best_min_samples = None
    best_silhouette_score = -1
    best_labels = None

    tanimoto_matrix = tanimoto_similarity_matrix(fingerprints)
    distance_matrix = 1 - tanimoto_matrix

    print("Inizio ricerca degli iperparametri per DBSCAN")
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
            labels = dbscan.fit_predict(distance_matrix)

            # Calcola l'indice di silhouette solo se ci sono almeno due cluster
            if len(set(labels)) > 1:
                silhouette_avg = silhouette_score(distance_matrix, labels, metric="precomputed")
                print(f"eps: {eps}, min_samples: {min_samples}, Silhouette Score: {silhouette_avg}")

                # Mantieni traccia della migliore combinazione di iperparametri
                if silhouette_avg > best_silhouette_score:
                    best_eps = eps
                    best_min_samples = min_samples
                    best_silhouette_score = silhouette_avg
                    best_labels = labels

    print(f"Migliore eps: {best_eps}, Migliore min_samples: {best_min_samples}, Silhouette Score: {best_silhouette_score}")
    return best_eps, best_min_samples, best_labels

if __name__ == "__main__":
    print("Avvio del programma principale")
    output_excel_drug = "fragment_analysis/TMMC/conc_fp_fragments_clusters_drug.xlsx"
    output_excel_non_drug = "fragment_analysis/TMMC/conc_fp_fragments_clusters_non_drug.xlsx"
    image_folder_drug = "fragment_analysis/TMMC/conc_fp_images_drug"
    image_folder_non_drug = "fragment_analysis/TMMC/conc_fp_images_non_drug"
    fragments_path = "high_attention_frags.csv"
    
    # Imposta il numero di worker per l'elaborazione parallela (regola in base al tuo CPU)
    max_workers = os.cpu_count() or 4
    print(f"Numero di processi worker impostati: {max_workers}")
    
    errors = analyze_fragments(
        fragments_path, 
        output_excel_drug, 
        output_excel_non_drug, 
        image_folder_drug, 
        image_folder_non_drug, 
        show=False,
        max_workers=max_workers
    )
    print(f"Numero totale di errori durante l'analisi: {errors}")
    print("Programma terminato")
