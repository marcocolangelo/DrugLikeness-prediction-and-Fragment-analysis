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
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import rdFingerprintGenerator
from sklearn.decomposition import PCA

# Definisci le dimensioni delle fingerprint
PER_FP_BITS_ECFP = 256
PER_FP_BITS_FCFP = 32
TOTAL_BITS = PER_FP_BITS_ECFP + PER_FP_BITS_FCFP  # 1126 bit

# Initialize MorganGenerator globally to avoid re-initializing in each process
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=PER_FP_BITS_ECFP)

# Funzione per calcolare le fingerprint
def compute_fingerprint(smiles, per_fp_bits_ecfp=PER_FP_BITS_ECFP, per_fp_bits_fcfp=PER_FP_BITS_FCFP):
    """
    Compute both ECFP and FCFP fingerprints and concatenate them.

    Parameters:
    - smiles (str): SMILES string of the molecule.
    - per_fp_bits_ecfp (int): Number of bits for ECFP.
    - per_fp_bits_fcfp (int): Number of bits for FCFP.

    Returns:
    - str: Concatenated bit string of ECFP and FCFP.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        # Calcolo ECFP (Morgan Fingerprint) utilizzando MorganGenerator
        ecfp = morgan_gen.GetFingerprint(mol)
        ecfp_str = ecfp.ToBitString()
    except Exception as e:
        print(f"Errore nella generazione di ECFP per SMILES {smiles}: {e}")
        return None

    try:
        # Calcolo FCFP (RDKFingerprint) con dimensione ridotta
        fcfp = Chem.RDKFingerprint(mol, fpSize=per_fp_bits_fcfp)
        fcfp_str = fcfp.ToBitString()
    except Exception as e:
        print(f"Errore nella generazione di FCFP per SMILES {smiles}: {e}")
        return None

    # Concatenazione delle due fingerprint
    combined_fp_str = ecfp_str + fcfp_str

    # Verifica della lunghezza
    expected_length = per_fp_bits_ecfp + per_fp_bits_fcfp
    if len(combined_fp_str) != expected_length:
        print(f"Errore: La fingerprint concatenata per SMILES {smiles} ha una lunghezza di {len(combined_fp_str)} invece di {expected_length}")
        return None

    return combined_fp_str

# Converti le fingerprint in array NumPy
def fingerprints_to_numpy(fingerprints):
    """
    Convert list of concatenated fingerprint bit strings to NumPy array.

    Parameters:
    - fingerprints (list of str): List of concatenated bit strings.

    Returns:
    - np.ndarray: 2D array of bits.
    - list of str: List of concatenated bit strings.
    """
    # Filtra fingerprint None
    valid_fps = [fp for fp in fingerprints if fp is not None]
    if not valid_fps:
        return np.array([]), []
    # Converti a array NumPy di 0 e 1
    fps_np = np.array([[int(bit) for bit in fp_str] for fp_str in valid_fps], dtype=np.uint8)
    return fps_np, valid_fps

# Funzione per estrarre le fingerprint usando parallelismo
def extract_fingerprints(df, per_fp_bits_ecfp=PER_FP_BITS_ECFP, per_fp_bits_fcfp=PER_FP_BITS_FCFP):
    """
    Extract and process fingerprints from DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'smiles', 'pred', and 'id_molecola'.
    - per_fp_bits_ecfp (int): Number of bits for ECFP.
    - per_fp_bits_fcfp (int): Number of bits for FCFP.

    Returns:
    - Various NumPy arrays and counts related to drug and non-drug fragments.
    """
    smiles_list = df['smiles'].tolist()
    preds = df['pred'].tolist()
    ids = df['id_molecola'].tolist()

    # Calcolo parallelo delle fingerprint con progress bar
    with Pool(cpu_count()) as pool:
        fps = list(tqdm(pool.imap(partial(compute_fingerprint, per_fp_bits_ecfp=per_fp_bits_ecfp, per_fp_bits_fcfp=per_fp_bits_fcfp), smiles_list), 
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
    """
    Deduplicate fingerprints and count occurrences.

    Parameters:
    - fps_np (np.ndarray): 2D array of fingerprints.
    - smiles (np.ndarray): Array of SMILES strings.
    - ids (np.ndarray): Array of molecule IDs.

    Returns:
    - defaultdict: Dictionary with deduplicated fingerprints.
    """
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
def dedup_dict_to_dense_matrix(dedup_dict, n_bits=TOTAL_BITS):
    """
    Convert deduplicated fingerprint dictionary to dense NumPy matrix.

    Parameters:
    - dedup_dict (defaultdict): Deduplicated fingerprint dictionary.
    - n_bits (int): Total number of bits per fingerprint.

    Returns:
    - np.ndarray: 2D array of fingerprints.
    - list of str: List of unique fingerprint bit strings.
    """
    unique_fps = list(dedup_dict.keys())
    num_fps = len(unique_fps)
    matrix = np.zeros((num_fps, n_bits), dtype=int)
    
    for i, fp_str in enumerate(unique_fps):
        if len(fp_str) != n_bits:
            print(f"Errore: La fingerprint alla posizione {i} ha una lunghezza di {len(fp_str)} invece di {n_bits}")
            continue  # Salta questa fingerprint o gestisci l'errore come preferisci
        on_bits = [idx for idx, bit in enumerate(fp_str) if bit == '1']
        # Verifica che tutti gli indici siano validi
        invalid_bits = [bit for bit in on_bits if bit >= n_bits or bit < 0]
        if invalid_bits:
            print(f"Errore: Fingerprint alla posizione {i} ha bit invalidi: {invalid_bits}")
            continue  # Salta questa fingerprint o gestisci l'errore come preferisci
        matrix[i, on_bits] = 1
    
    return matrix, unique_fps

# Funzione per applicare PCA
def apply_pca_reduction(fps_np, n_components=100):
    """
    Apply PCA to reduce the dimensionality of fingerprint data.

    Parameters:
    - fps_np (np.ndarray): 2D array of fingerprints.
    - n_components (int): Number of principal components.

    Returns:
    - np.ndarray: Reduced fingerprint data.
    """
    pca = PCA(n_components=n_components, random_state=42)
    reduced_fps = pca.fit_transform(fps_np)
    return reduced_fps

# Funzione per eseguire HDBSCAN Clustering
def hdbscan_clustering(fps, min_cluster_size=10, metric='euclidean'):
    """
    Perform HDBSCAN clustering on fingerprint data.

    Parameters:
    - fps (np.ndarray): 2D array of fingerprints.
    - min_cluster_size (int): Minimum cluster size for HDBSCAN.
    - metric (str): Distance metric for clustering.

    Returns:
    - np.ndarray: Cluster labels for each fingerprint.
    """
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric=metric)
    labels = clusterer.fit_predict(fps)
    return labels

# Funzione per salvare immagini delle molecole
def save_molecule_images(smiles_list, labels, image_folder):
    """
    Save images of molecules with cluster information.

    Parameters:
    - smiles_list (list of str): List of SMILES strings.
    - labels (np.ndarray): Cluster labels.
    - image_folder (str): Directory to save images.

    Returns:
    - list of str: Paths to saved images.
    """
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
    per_fp_bits_ecfp=PER_FP_BITS_ECFP, 
    per_fp_bits_fcfp=PER_FP_BITS_FCFP, 
    pca_components=100, 
    min_cluster_size=10
):
    """
    Main function to analyze fragments using HDBSCAN clustering.

    Parameters:
    - fragments_path (str): Path to the input CSV file.
    - output_excel_drug (str): Output Excel file for drug fragments.
    - output_excel_non_drug (str): Output Excel file for non-drug fragments.
    - image_folder_drug (str): Directory to save drug fragment images.
    - image_folder_non_drug (str): Directory to save non-drug fragment images.
    - per_fp_bits_ecfp (int): Number of bits for ECFP.
    - per_fp_bits_fcfp (int): Number of bits for FCFP.
    - pca_components (int): Number of components for PCA.
    - min_cluster_size (int): Minimum cluster size for HDBSCAN.

    Returns:
    - int: Number of errors encountered during fingerprint computation.
    """
    df = pd.read_csv(fragments_path)

    # Estrazione delle fingerprint
    fp_drug, fp_non_drug, smiles_drug, smiles_non_drug, id_drug, id_non_drug, errors = extract_fingerprints(df, per_fp_bits_ecfp=per_fp_bits_ecfp, per_fp_bits_fcfp=per_fp_bits_fcfp)
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
        dense_drug, unique_fps_drug = dedup_dict_to_dense_matrix(dedup_drug, n_bits=TOTAL_BITS)
    else:
        dense_drug = np.array([])
        unique_fps_drug = []
    
    if dedup_non_drug:
        dense_non_drug, unique_fps_non_drug = dedup_dict_to_dense_matrix(dedup_non_drug, n_bits=TOTAL_BITS)
    else:
        dense_non_drug = np.array([])
        unique_fps_non_drug = []

    # Applicare PCA per ridurre la dimensionalità
    if dense_drug.size > 0:
        print("Applica PCA ai Drug Fragments...")
        reduced_drug = apply_pca_reduction(dense_drug, n_components=pca_components)
    else:
        reduced_drug = np.array([])
        print("Nessun dato drug da ridurre.")

    if dense_non_drug.size > 0:
        print("Applica PCA ai Non-Drug Fragments...")
        reduced_non_drug = apply_pca_reduction(dense_non_drug, n_components=pca_components)
    else:
        reduced_non_drug = np.array([])
        print("Nessun dato non-drug da ridurre.")

    # Clustering per il gruppo drug
    if reduced_drug.size > 0:
        print("Clustering Drug Fragments...")
        drug_labels = hdbscan_clustering(reduced_drug, min_cluster_size=min_cluster_size, metric='euclidean')
        num_clusters_drug = len(set(drug_labels)) - (1 if -1 in drug_labels else 0)
        print(f"Drug Clusters Found: {num_clusters_drug}")
        drug_image_paths = save_molecule_images([dedup_drug[fp]['smi'] for fp in unique_fps_drug], drug_labels, image_folder_drug)
    else:
        drug_labels = []
        drug_image_paths = []
        print("Nessun cluster drug trovato.")

    # Clustering per il gruppo non-drug
    if reduced_non_drug.size > 0:
        print("Clustering Non-Drug Fragments...")
        non_drug_labels = hdbscan_clustering(reduced_non_drug, min_cluster_size=min_cluster_size, metric='euclidean')
        num_clusters_non_drug = len(set(non_drug_labels)) - (1 if -1 in non_drug_labels else 0)
        print(f"Non-Drug Clusters Found: {num_clusters_non_drug}")
        non_drug_image_paths = save_molecule_images([dedup_non_drug[fp]['smi'] for fp in unique_fps_non_drug], non_drug_labels, image_folder_non_drug)
    else:
        non_drug_labels = []
        non_drug_image_paths = []
        print("Nessun cluster non-drug trovato.")

    # Funzione per salvare i dati in Excel considerando le duplicazioni
    def save_to_excel(output_excel, dedup_dict, unique_fps, labels, image_paths):
        """
        Save clustering results to an Excel file.

        Parameters:
        - output_excel (str): Path to the output Excel file.
        - dedup_dict (defaultdict): Deduplicated fingerprint dictionary.
        - unique_fps (list of str): List of unique fingerprint bit strings.
        - labels (np.ndarray): Cluster labels.
        - image_paths (list of str): Paths to molecule images.
        """
        wb = Workbook()
        ws = wb.active
        ws.title = "Fragments Clusters"
        ws.append(['SMILES', 'Cluster', 'ID Molecola', 'Count', 'Image'])

        for i, (fp_str, label, img_path) in enumerate(zip(unique_fps, labels, image_paths)):
            smi = dedup_dict[fp_str]['smi']
            mol_ids = dedup_dict[fp_str]['ids']
            count = dedup_dict[fp_str]['count']
            mol_ids_str = ', '.join(map(str, mol_ids))
            ws.append([smi, label, mol_ids_str, count])
            if img_path:
                try:
                    img = ExcelImage(img_path)
                    img.height = 100
                    img.width = 100
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
    output_excel_drug = "data/fragment_analysis/TMMC/conc_fp_fragments_clusters_drug.xlsx"
    output_excel_non_drug = "data/fragment_analysis/TMMC/conc_fp_fragments_clusters_non_drug.xlsx"
    image_folder_drug = "data/fragment_analysis/TMMC/conc_fp_images_drug"
    image_folder_non_drug = "data/fragment_analysis/TMMC/conc_fp_images_non_drug"
    fragments_path = "data/test_data/TCMBANK/high_attention_frags.csv"

    # Esegui l'analisi
    errors = analyze_fragments_hdbscan(
        fragments_path=fragments_path,
        output_excel_drug=output_excel_drug,
        output_excel_non_drug=output_excel_non_drug,
        image_folder_drug=image_folder_drug,
        image_folder_non_drug=image_folder_non_drug,
        per_fp_bits_ecfp=PER_FP_BITS_ECFP,  # 1024 bit per ECFP
        per_fp_bits_fcfp=PER_FP_BITS_FCFP,  # 102 bit per FCFP
        pca_components=100,                # Numero di componenti per PCA
        min_cluster_size=5                # Puoi regolare questo parametro in base alle tue esigenze
    )

    print(f"Errors: {errors}")
