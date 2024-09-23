import hashlib
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
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

def compute_extended_fp(smiles, num_neigh, n_bits=512):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Error: Unable to parse SMILES: {smiles}")
        return None
    
    ecfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    fcfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, useFeatures=True, nBits=n_bits)
    ecfp4_numpy = np.array([int(bit) for bit in ecfp4])
    ecfp4_numpy_weighted = np.tile(ecfp4_numpy, 2)
    ecfp4_weighted = numpy_to_bitvect(ecfp4_numpy_weighted)
    concatenated_fingerprint = concatenate_fingerprints(ecfp4_weighted, fcfp4)
    # concatenated_fingerprint = concatenate_fingerprints(ecfp4, fcfp4)
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
def perform_tanimoto_clustering(fingerprints, dataset,show = False):

    labels, cluster_counts,tanimoto_matrix = dbscan_clustering(fingerprints, eps=0.6, min_samples=2)

    mean_similarity = calculate_average_tanimoto_similarity(tanimoto_matrix)
    print(f"Mean Tanimoto similarity for dataset {dataset}: {mean_similarity}")

    if show == True:
        plt.imshow(tanimoto_matrix, cmap='hot', interpolation='nearest')
        plt.title(f"Matrice di Similarità di Tanimoto for {dataset} dataset")
        plt.colorbar()
        plt.savefig(f"data/fragment_analysis/tanimoto_similarity_matrix_{dataset}.png")
        plt.show()

    
    return labels, cluster_counts

def dbscan_clustering(fingerprints, eps=0.3, min_samples=2):
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
    print(f"Cluster counts: {cluster_counts}")
    return labels, cluster_counts,tanimoto_matrix

# Funzione per salvare le immagini dei frammenti
def save_smiles_images(smiles_list, labels, image_folder):
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    
    image_paths = []
    for i, (smiles, cluster) in enumerate(zip(smiles_list, labels)):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Error: Unable to generate molecule from SMILES: {smiles}")
            continue
        
        img = Draw.MolToImage(mol, size=(300, 300))
        img_path = os.path.join(image_folder, f"fragment_{i}_cluster_{cluster}.png")
        img.save(img_path)
        image_paths.append(img_path)
    
    return image_paths

def extract_fp(df,n_bits=1024):
    fp_drug = []
    fp_non_drug = []
    smiles_drug = []
    smiles_non_drug = []
    id_drug = []
    id_non_drug = []
    real_label_drug = []
    real_label_non_drug = []
    errors = 0

    for i, row in df.iterrows():
        smiles = row["smiles"]
        num_neigh = row["num_neigh"]
        pred = row["pred"]
        mol_id = row["id_molecola"]  # Ottieni l'ID della molecola
        # real_label = row["real_label"]  # Ottieni il real_label
        real_label = None
        fp = compute_extended_fp(smiles, num_neigh, n_bits=n_bits)

        if fp is None:
            errors += 1
            continue

        if pred == 1:
            fp_drug.append(fp)
            smiles_drug.append(smiles)
            id_drug.append(mol_id)
            real_label_drug.append(real_label)
        else:
            fp_non_drug.append(fp)
            smiles_non_drug.append(smiles)
            id_non_drug.append(mol_id)
            real_label_non_drug.append(real_label)

    return fp_drug, fp_non_drug, smiles_drug, smiles_non_drug, id_drug, id_non_drug, real_label_drug, real_label_non_drug,errors

# Funzione aggiornata per includere l'ID molecola e la real_label
def analyze_fragments(fragments_path, output_excel_drug, output_excel_non_drug, image_folder_drug, image_folder_non_drug,show = False):
    
    df = pd.read_csv(fragments_path)

    fp_drug, fp_non_drug, smiles_drug, smiles_non_drug, id_drug, id_non_drug, real_label_drug, real_label_non_drug,errors = extract_fp(df,n_bits=1024)

    print(f"Drug Fragments fp len: {len(fp_drug[0])}")
    print(f"Non-Drug Fragments fp len: {len(fp_non_drug[0])}")


    # Clustering per il gruppo drug
    drug_labels, _ = perform_tanimoto_clustering(fp_drug, "drug",show=show)
    drug_image_paths = save_smiles_images(smiles_drug, drug_labels, image_folder_drug)

    # Clustering per il gruppo non-drug
    non_drug_labels, _ = perform_tanimoto_clustering(fp_non_drug, "non_drug",show=show)
    non_drug_image_paths = save_smiles_images(smiles_non_drug, non_drug_labels, image_folder_non_drug)

    # Salva dati per il gruppo drug in Excel
    wb_drug = Workbook()
    ws_drug = wb_drug.active
    ws_drug.title = "Fragments Clusters"

    ws_drug.append(['SMILES', 'Cluster', 'ID Molecola', 'Real Label', 'Image'])  # Aggiungi "Real Label"

    for i, (smiles, cluster, mol_id, real_label, img_path) in enumerate(zip(smiles_drug, drug_labels, id_drug, real_label_drug, drug_image_paths)):
        ws_drug.append([smiles, cluster, mol_id, real_label])
        img = Image(img_path)
        img.height = 100
        img.width = 100
        ws_drug.add_image(img, f"E{i+2}")

    wb_drug.save(output_excel_drug)

    # Salva dati per il gruppo non-drug in Excel
    wb_non_drug = Workbook()
    ws_non_drug = wb_non_drug.active
    ws_non_drug.title = "Fragments Clusters"

    ws_non_drug.append(['SMILES', 'Cluster', 'ID Molecola', 'Real Label', 'Image'])  # Aggiungi "Real Label"

    for i, (smiles, cluster, mol_id, real_label, img_path) in enumerate(zip(smiles_non_drug, non_drug_labels, id_non_drug, real_label_non_drug, non_drug_image_paths)):
        ws_non_drug.append([smiles, cluster, mol_id, real_label])
        img = Image(img_path)
        img.height = 100
        img.width = 100
        ws_non_drug.add_image(img, f"E{i+2}")

    wb_non_drug.save(output_excel_non_drug)

    # Crea riepilogo cluster per il gruppo drug
    cluster_summary_drug = pd.DataFrame({
        'Cluster': list(set(drug_labels)),
        'Num_Elements': list(Counter(drug_labels).values()),
        'Num_Unique_SMILES': [len(set(smiles_drug)) for _ in set(drug_labels)]
    })

    # Crea riepilogo cluster per il gruppo non-drug
    cluster_summary_non_drug = pd.DataFrame({
        'Cluster': list(set(non_drug_labels)),
        'Num_Elements': list(Counter(non_drug_labels).values()),
        'Num_Unique_SMILES': [len(set(smiles_non_drug)) for _ in set(non_drug_labels)]
    })

    # Aggiungi riepilogo cluster al file drug
    with pd.ExcelWriter(output_excel_drug, engine='openpyxl', mode='a') as writer_drug:
        cluster_summary_drug.to_excel(writer_drug, sheet_name="Cluster Summary", index=False)

    # Aggiungi riepilogo cluster al file non-drug
    with pd.ExcelWriter(output_excel_non_drug, engine='openpyxl', mode='a') as writer_non_drug:
        cluster_summary_non_drug.to_excel(writer_non_drug, sheet_name="Cluster Summary", index=False)

    return errors



if __name__ == "__main__":
    output_excel_drug = "data/fragment_analysis/TMMC/deep_search_fragments_clusters_drug.xlsx"
    output_excel_non_drug = "data/fragment_analysis/TMMC/deep_search_fragments_clusters_non_drug.xlsx"
    image_folder_drug = "data/fragment_analysis/TMMC/deep_search_images_drug"
    image_folder_non_drug = "data/fragment_analysis/TMMC/deep_search_images_non_drug"
    fragments_path = "data/test_data/TCMBANK/TMMC_deep_search_high_attention_frags.csv"
    errors = analyze_fragments(fragments_path, output_excel_drug, output_excel_non_drug, image_folder_drug, image_folder_non_drug,show = False)
    print(f"Errors: {errors}")

    

def dbscan_hyperparameter_search(fingerprints, eps_values, min_samples_values):
    
    best_eps = None
    best_min_samples = None
    best_silhouette_score = -1
    best_labels = None

    tanimoto_matrix = tanimoto_similarity_matrix(fingerprints)
    distance_matrix = 1 - tanimoto_matrix

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

