import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import hdbscan
from openpyxl import Workbook
from openpyxl.drawing.image import Image as ExcelImage
from collections import Counter
from functools import partial
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map, thread_map

# Funzione per calcolare le fingerprint
def compute_fingerprint(smiles, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return fp

# Funzione per convertire le fingerprint in stringhe (per deduplicazione)
def fingerprint_to_str(fp):
    return fp.ToBitString() if fp is not None else None

# Funzione per deduplicare le fingerprint e contare le occorrenze
def deduplicate_fingerprints(fps, smiles, ids):
    dedup_dict = {}
    for fp, smi, mol_id in zip(fps, smiles, ids):
        if fp is None:
            continue
        fp_str = fingerprint_to_str(fp)
        if fp_str in dedup_dict:
            dedup_dict[fp_str]['count'] += 1
            dedup_dict[fp_str]['ids'].append(mol_id)
        else:
            dedup_dict[fp_str] = {
                'count': 1,
                'smi': smi,
                'ids': [mol_id]
            }
    return dedup_dict

# Funzione per convertire fingerprint stringhe in matrici dense
def dedup_dict_to_dense_matrix(dedup_dict, n_bits=1024):
    unique_fps = list(dedup_dict.keys())
    num_fps = len(unique_fps)
    matrix = np.zeros((num_fps, n_bits), dtype=int)
    
    for i, fp_str in enumerate(unique_fps):
        on_bits = [idx for idx, bit in enumerate(fp_str) if bit == '1']
        matrix[i, on_bits] = 1
    
    return matrix, unique_fps

# Funzione per estrarre e deduplicare le fingerprint
def extract_and_deduplicate_fingerprints(df, n_bits=1024):
    smiles_list = df['smiles'].tolist()
    preds = df['pred'].tolist()
    ids = df['id_molecola'].tolist()

    # Calcolo parallelo delle fingerprint con progress bar
    fps = process_map(
        partial(compute_fingerprint, n_bits=n_bits),
        smiles_list,
        max_workers=os.cpu_count(),
        chunksize=100,  # Regola in base alle tue esigenze
        desc="Calcolando le fingerprint"
    )

    # Separazione in gruppi drug e non-drug utilizzando list comprehension
    fp_drug = [fp for fp, pred in zip(fps, preds) if pred == 1]
    smiles_drug = [smi for smi, pred in zip(smiles_list, preds) if pred == 1]
    id_drug = [mol_id for mol_id, pred in zip(ids, preds) if pred == 1]

    fp_non_drug = [fp for fp, pred in zip(fps, preds) if pred == 0]
    smiles_non_drug = [smi for smi, pred in zip(smiles_list, preds) if pred == 0]
    id_non_drug = [mol_id for mol_id, pred in zip(ids, preds) if pred == 0]

    # Deduplicazione
    dedup_drug = deduplicate_fingerprints(fp_drug, smiles_drug, id_drug)
    dedup_non_drug = deduplicate_fingerprints(fp_non_drug, smiles_non_drug, id_non_drug)

    return dedup_drug, dedup_non_drug

# Funzione per eseguire HDBSCAN Clustering su matrici dense
def hdbscan_clustering_dense(dense_fps, min_cluster_size=3, metric='jaccard'):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric=metric, memory='disk')
    labels = clusterer.fit_predict(dense_fps)
    return labels

# Funzione per salvare immagini delle molecole
def save_image(mol, img_path):
    if mol is None:
        return None
    Draw.MolToFile(mol, img_path, size=(200, 200))
    return img_path

def save_molecule_images_parallel(unique_fps, dedup_dict, labels, image_folder, max_workers=8):
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    
    # Funzione di salvataggio con gestione delle immagini
    def save_image_task(args):
        i, fp_str, label = args
        smi = dedup_dict[fp_str]['smi']
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        img_filename = f"unique_fragment_{i}_cluster_{label}.png"
        img_path = os.path.join(image_folder, img_filename)
        try:
            Draw.MolToFile(mol, img_path, size=(200, 200))
            return img_path
        except Exception as e:
            print(f"Errore nel salvataggio dell'immagine per SMILES {smi}: {e}")
            return None
    
    # Creare una lista di task da eseguire
    tasks = [(i, fp_str, label) for i, (fp_str, label) in enumerate(zip(unique_fps, labels))]
    
    # Parallelizzare il salvataggio delle immagini con progress bar
    image_paths = thread_map(
        save_image_task,
        tasks,
        max_workers=max_workers,
        desc=f"Salvataggio immagini in {image_folder}",
        chunksize=100  # Regola in base alle tue esigenze
    )
    
    return image_paths

# Funzione principale per analizzare i frammenti con deduplicazione
def analyze_fragments_deduplicated(
    fragments_path, 
    output_excel_drug, 
    output_excel_non_drug, 
    image_folder_drug, 
    image_folder_non_drug, 
    n_bits=1024, 
    min_cluster_size=10
):
    # Leggi il dataset
    df = pd.read_csv(fragments_path)
    
    # Estrazione e deduplicazione delle fingerprint
    dedup_drug, dedup_non_drug = extract_and_deduplicate_fingerprints(df, n_bits=n_bits)
    print(f"Unique Drug Fingerprints: {len(dedup_drug)}, Unique Non-Drug Fingerprints: {len(dedup_non_drug)}")
    
    # Converti i dizionari deduplicati in matrici dense
    dense_drug, unique_fps_drug = dedup_dict_to_dense_matrix(dedup_drug, n_bits=n_bits)
    dense_non_drug, unique_fps_non_drug = dedup_dict_to_dense_matrix(dedup_non_drug, n_bits=n_bits)
    
    # Clustering per il gruppo drug
    print("Clustering Drug Fragments...")
    drug_labels = hdbscan_clustering_dense(dense_drug, min_cluster_size=min_cluster_size, metric='jaccard')
    num_clusters_drug = len(set(drug_labels)) - (1 if -1 in drug_labels else 0)
    print(f"Drug Clusters Found: {num_clusters_drug}")
    drug_image_paths = save_molecule_images_parallel(unique_fps_drug, dedup_drug, drug_labels, image_folder_drug)
    
    # Clustering per il gruppo non-drug
    print("Clustering Non-Drug Fragments...")
    non_drug_labels = hdbscan_clustering_dense(dense_non_drug, min_cluster_size=min_cluster_size, metric='jaccard')
    num_clusters_non_drug = len(set(non_drug_labels)) - (1 if -1 in non_drug_labels else 0)
    print(f"Non-Drug Clusters Found: {num_clusters_non_drug}")
    non_drug_image_paths = save_molecule_images_parallel(unique_fps_non_drug, dedup_non_drug, non_drug_labels, image_folder_non_drug)
    
    # Funzione per salvare i dati in Excel considerando le duplicazioni
    def save_to_excel(output_excel, dedup_dict, unique_fps, labels, image_paths):
        wb = Workbook()
        ws = wb.active
        ws.title = "Fragments Clusters"
        ws.append(['SMILES', 'Cluster', 'ID Molecola', 'Count', 'Image'])
    
        for i, (fp_str, label, img_path) in enumerate(tqdm(zip(unique_fps, labels, image_paths), total=len(unique_fps), desc=f"Salvataggio dati in {output_excel}")):
            smi = dedup_dict[fp_str]['smi']
            mol_ids = dedup_dict[fp_str]['ids']
            count = dedup_dict[fp_str]['count']
            # Per semplificare, salvo solo il primo ID
            first_id = mol_ids[0] if mol_ids else None
            ws.append([smi, label, first_id, count])
            if img_path:
                try:
                    img = ExcelImage(img_path)
                    img.height = 100
                    img.width = 100
                    ws.add_image(img, f"E{i+2}")
                except Exception as e:
                    print(f"Errore nell'aggiunta dell'immagine {img_path} all'Excel: {e}")
    
        # Crea riepilogo cluster
        cluster_counts = Counter(labels)
        cluster_summary = pd.DataFrame({
            'Cluster': list(cluster_counts.keys()),
            'Num_Elements': list(cluster_counts.values())
        })
    
        # Aggiungi il riepilogo cluster
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            wb.save(output_excel)
            cluster_summary.to_excel(writer, sheet_name="Cluster Summary", index=False)
    
        return
    
    # Salva i dati in Excel
    print("Saving Drug Cluster Data to Excel...")
    save_to_excel(output_excel_drug, dedup_drug, unique_fps_drug, drug_labels, drug_image_paths)
    
    print("Saving Non-Drug Cluster Data to Excel...")
    save_to_excel(output_excel_non_drug, dedup_non_drug, unique_fps_non_drug, non_drug_labels, non_drug_image_paths)
    
    # Calcola il totale degli errori
    errors = (len(df) - len(dense_drug)) + (len(df) - len(dense_non_drug))
    return errors

if __name__ == "__main__":
    # Definisci i percorsi dei file e delle cartelle
    output_excel_drug = "data/fragment_analysis/TMMC/deep_search_fragments_clusters_drug.xlsx"
    output_excel_non_drug = "data/fragment_analysis/TMMC/deep_search_fragments_clusters_non_drug.xlsx"
    image_folder_drug = "data/fragment_analysis/TMMC/deep_search_images_drug"
    image_folder_non_drug = "data/fragment_analysis/TMMC/deep_search_images_non_drug"
    fragments_path = "data/test_data/TCMBANK/TMMC_deep_search_high_attention_frags.csv"
    
    # Esegui l'analisi
    errors = analyze_fragments_deduplicated(
        fragments_path=fragments_path,
        output_excel_drug=output_excel_drug,
        output_excel_non_drug=output_excel_non_drug,
        image_folder_drug=image_folder_drug,
        image_folder_non_drug=image_folder_non_drug,
        n_bits=1024,          # Puoi ridurre a 512 o 256 se necessario
        min_cluster_size=3   # Puoi regolare questo parametro in base alle tue esigenze
    )
    
    print(f"Errors: {errors}")
