import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import hdbscan
from openpyxl import Workbook
from openpyxl.drawing.image import Image as ExcelImage
from openpyxl.utils.dataframe import dataframe_to_rows
from collections import defaultdict, Counter
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# Funzione per calcolare le fingerprint
def compute_fingerprint(smiles, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return fp

# Converti le fingerprint in array NumPy
def fingerprints_to_numpy(fingerprints):
    # Filtra fingerprint None
    valid_fps = [fp for fp in fingerprints if fp is not None]
    if not valid_fps:
        return np.array([]), []
    # Converti a stringhe per facilitare la deduplicazione
    fp_strs = [fp.ToBitString() for fp in valid_fps]
    # Converti a array NumPy di 0 e 1
    fps_np = np.array([[int(bit) for bit in fp_str] for fp_str in fp_strs], dtype=np.uint8)
    return fps_np, fp_strs

# Funzione per estrarre le fingerprint usando parallelismo
def extract_fingerprints(df, n_bits=1024):
    smiles_list = df['smiles'].tolist()
    preds = df['pred'].tolist()
    ids = df['id_molecola'].tolist()

    # Calcolo parallelo delle fingerprint con progress bar
    with Pool(cpu_count()) as pool:
        fps = list(tqdm(pool.imap(partial(compute_fingerprint, n_bits=n_bits), smiles_list), 
                        total=len(smiles_list), desc="Calcolando le fingerprint"))

    # Converti le fingerprint in NumPy e ottieni le stringhe corrispondenti
    fps_np, fp_strs = fingerprints_to_numpy(fps)
    
    # Filtra fingerprint non valide (all-zero)
    if fps_np.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), len(fps)
    
    valid_indices = np.any(fps_np, axis=1)
    fps_np = fps_np[valid_indices]
    smiles_valid = np.array(smiles_list)[valid_indices]
    preds = np.array(preds)[valid_indices]
    ids = np.array(ids)[valid_indices]

    # Separazione in drug e non-drug
    fp_drug = fps_np[preds == 1]
    smiles_drug = smiles_valid[preds == 1]
    id_drug = ids[preds == 1]

    fp_non_drug = fps_np[preds == 0]
    smiles_non_drug = smiles_valid[preds == 0]
    id_non_drug = ids[preds == 0]

    errors = len(fps) - len(fps_np)

    return fp_drug, fp_non_drug, smiles_drug, smiles_non_drug, id_drug, id_non_drug, errors

# Funzione per deduplicare le fingerprint e contare le occorrenze
def deduplicate_fingerprints(fps_np, smiles, ids):
    dedup_dict = defaultdict(lambda: {'smi': None, 'ids': [], 'count': 0})
    
    for fp, smi, mol_id in zip(fps_np, smiles, ids):
        # Converti la fingerprint in una stringa per usarla come chiave
        fp_str = ''.join(fp.astype(str))
        if dedup_dict[fp_str]['count'] == 0:
            dedup_dict[fp_str]['smi'] = smi
        dedup_dict[fp_str]['ids'].append(mol_id)
        dedup_dict[fp_str]['count'] += 1
    
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

# Funzione per eseguire HDBSCAN Clustering
def hdbscan_clustering(fps, min_cluster_size=10, metric='jaccard'):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric=metric)
    labels = clusterer.fit_predict(fps)
    return labels

# Funzione per salvare immagini delle molecole
def save_molecule_images(smiles_list, labels, image_folder):
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    image_paths = []
    for i, (smiles, cluster) in enumerate(zip(smiles_list, labels)):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            image_paths.append(None)
            continue
        img_path = os.path.join(image_folder, f"fragment_{i}_cluster_{cluster}.png")
        Draw.MolToFile(mol, img_path, size=(200, 200))
        image_paths.append(img_path)
    return image_paths

# Funzione principale per analizzare i frammenti
def analyze_fragments_hdbscan(
    fragments_path, 
    output_excel_drug, 
    output_excel_non_drug, 
    image_folder_drug, 
    image_folder_non_drug, 
    n_bits=1024, 
    min_cluster_size=10
):
    df = pd.read_csv(fragments_path)

    # Estrazione delle fingerprint
    fp_drug, fp_non_drug, smiles_drug, smiles_non_drug, id_drug, id_non_drug, errors = extract_fingerprints(df, n_bits=n_bits)
    print(f"Drug Fragments: {fp_drug.shape[0]}, Non-Drug Fragments: {fp_non_drug.shape[0]}, Errors: {errors}")

    if fp_drug.size == 0 and fp_non_drug.size == 0:
        print("Nessuna fingerprint valida trovata. Interruzione del processo.")
        return errors

    # Deduplicazione per il gruppo drug
    if fp_drug.size > 0:
        print("Deduplicazione Drug Fragments...")
        dedup_drug = deduplicate_fingerprints(fp_drug, smiles_drug, id_drug)
        print(f"Unique Drug Fingerprints: {len(dedup_drug)}")
    else:
        dedup_drug = {}
        print("Nessun frammento drug valido trovato.")

    # Deduplicazione per il gruppo non-drug
    if fp_non_drug.size > 0:
        print("Deduplicazione Non-Drug Fragments...")
        dedup_non_drug = deduplicate_fingerprints(fp_non_drug, smiles_non_drug, id_non_drug)
        print(f"Unique Non-Drug Fingerprints: {len(dedup_non_drug)}")
    else:
        dedup_non_drug = {}
        print("Nessun frammento non-drug valido trovato.")

    # Converti i dizionari deduplicati in matrici dense
    if dedup_drug:
        dense_drug, unique_fps_drug = dedup_dict_to_dense_matrix(dedup_drug, n_bits=n_bits)
    else:
        dense_drug = np.array([])
        unique_fps_drug = []
    
    if dedup_non_drug:
        dense_non_drug, unique_fps_non_drug = dedup_dict_to_dense_matrix(dedup_non_drug, n_bits=n_bits)
    else:
        dense_non_drug = np.array([])
        unique_fps_non_drug = []

    # Clustering per il gruppo drug
    if dense_drug.size > 0:
        print("Clustering Drug Fragments...")
        drug_labels = hdbscan_clustering(dense_drug, min_cluster_size=min_cluster_size, metric='jaccard')
        num_clusters_drug = len(set(drug_labels)) - (1 if -1 in drug_labels else 0)
        print(f"Drug Clusters Found: {num_clusters_drug}")
        drug_image_paths = save_molecule_images([dedup_drug[fp]['smi'] for fp in unique_fps_drug], drug_labels, image_folder_drug)
    else:
        drug_labels = []
        drug_image_paths = []
        print("Nessun cluster drug trovato.")

    # Clustering per il gruppo non-drug
    if dense_non_drug.size > 0:
        print("Clustering Non-Drug Fragments...")
        non_drug_labels = hdbscan_clustering(dense_non_drug, min_cluster_size=min_cluster_size, metric='jaccard')
        num_clusters_non_drug = len(set(non_drug_labels)) - (1 if -1 in non_drug_labels else 0)
        print(f"Non-Drug Clusters Found: {num_clusters_non_drug}")
        non_drug_image_paths = save_molecule_images([dedup_non_drug[fp]['smi'] for fp in unique_fps_non_drug], non_drug_labels, image_folder_non_drug)
    else:
        non_drug_labels = []
        non_drug_image_paths = []
        print("Nessun cluster non-drug trovato.")

    # Funzione per salvare i dati in Excel considerando le duplicazioni
    def save_to_excel(output_excel, dedup_dict, unique_fps, labels, image_paths):
        wb = Workbook()
        ws = wb.active
        ws.title = "Fragments Clusters"
        ws.append(['SMILES', 'Cluster', 'ID Molecola', 'Count', 'Image'])

        for i, (fp_str, label, img_path) in enumerate(zip(unique_fps, labels, image_paths)):
            smi = dedup_dict[fp_str]['smi']
            mol_ids = dedup_dict[fp_str]['ids']
            count = dedup_dict[fp_str]['count']
            # Converti la lista di ID molecola in una stringa separata da virgole
            mol_ids_str = ', '.join(map(str, mol_ids))
            ws.append([smi, label, mol_ids_str, count])
            if img_path:
                try:
                    img = ExcelImage(img_path)
                    img.height = 100
                    img.width = 100
                    # Posiziona l'immagine nella colonna E (5) e nella riga i+2
                    cell = f"E{i+2}"
                    ws.add_image(img, cell)
                except Exception as e:
                    print(f"Errore nell'aggiunta dell'immagine {img_path} all'Excel: {e}")

        # Crea riepilogo cluster
        cluster_counts = Counter(labels)
        cluster_summary = pd.DataFrame({
            'Cluster': list(cluster_counts.keys()),
            'Num_Elements': list(cluster_counts.values())
        })

        # Aggiungi un nuovo foglio per "Cluster Summary"
        ws_summary = wb.create_sheet(title="Cluster Summary")
        for r in dataframe_to_rows(cluster_summary, index=False, header=True):
            ws_summary.append(r)

        # Salva il workbook
        wb.save(output_excel)

        return

    # Salva i dati in Excel per il gruppo drug
    if dedup_drug:
        print("Saving Drug Cluster Data to Excel...")
        save_to_excel(output_excel_drug, dedup_drug, unique_fps_drug, drug_labels, drug_image_paths)
    else:
        print("Nessun dato drug da salvare in Excel.")

    # Salva i dati in Excel per il gruppo non-drug
    if dedup_non_drug:
        print("Saving Non-Drug Cluster Data to Excel...")
        save_to_excel(output_excel_non_drug, dedup_non_drug, unique_fps_non_drug, non_drug_labels, non_drug_image_paths)
    else:
        print("Nessun dato non-drug da salvare in Excel.")

    return errors

if __name__ == "__main__":
    # Definisci i percorsi dei file e delle cartelle
    output_excel_drug = "clustering/TMMC/uf_fragments_clusters_drug.xlsx"
    output_excel_non_drug = "clustering/TMMC/uf_fragments_clusters_non_drug.xlsx"
    image_folder_drug = "clustering/TMMC/uf_images_drug"
    image_folder_non_drug = "clustering/TMMC/uf_images_non_drug"
    fragments_path = "high_attention_frags.csv"

    # Esegui l'analisi
    errors = analyze_fragments_hdbscan(
        fragments_path=fragments_path,
        output_excel_drug=output_excel_drug,
        output_excel_non_drug=output_excel_non_drug,
        image_folder_drug=image_folder_drug,
        image_folder_non_drug=image_folder_non_drug,
        n_bits=1024,          # Puoi ridurre a 512 o 256 se necessario
        min_cluster_size=3   # Puoi regolare questo parametro in base alle tue esigenze
    )

    print(f"Errors: {errors}")
