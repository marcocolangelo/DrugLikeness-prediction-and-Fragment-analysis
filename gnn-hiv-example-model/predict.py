import argparse
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader  # Aggiornato per usare il nuovo DataLoader
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from dataset_featurizer import MoleculeDataset
from model import GNN
import os
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from my_code.highlight_mol import highlight_mol_per_graph
from IPython.display import display

from PIL import Image
import io
import cairosvg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from openpyxl.utils.dataframe import dataframe_to_rows


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(filepath):
    print(f"Loading model from {filepath}")
    model = torch.load(filepath, map_location=device)
    model.to(device)
    print(f"Model loaded successfully.")
    return model

def calculate_metrics(y_pred, y_true, epoch, split):
    print(f"\n Confusion matrix: \n {confusion_matrix(y_pred, y_true)}")
    print(f"F1 Score: {f1_score(y_true, y_pred)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    try:
        roc = roc_auc_score(y_true, y_pred)
        print(f"ROC AUC: {roc}")
    except:
        print(f"ROC AUC: not defined")

def log_conf_matrix(y_pred, y_true, epoch,final = False):
    # Log confusion matrix as image
    cm = confusion_matrix(y_pred, y_true)
    classes = ["0", "1"]
    df_cfm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(10, 7))
    cfm_plot = sns.heatmap(df_cfm, annot=True, cmap='Blues', fmt='g')
    if final == True:
        cfm_plot.figure.savefig(f'data/test_data/TCMBANK/images/cm_{epoch}.png')
    else:
        cfm_plot.figure.savefig(f'data/test_data/images/cm_{epoch}.png')

def optimize_threshold(pred_probs, true_labels):
    thresholds = np.arange(0.0, 1.0, 0.01)
    best_f1 = 0
    best_threshold = 0.5

    for threshold in thresholds:
        preds = (pred_probs >= threshold).astype(int)
        f1 = f1_score(true_labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold


def test_model(model, test_data_path, batch_size, pos_weight,deep_search = True, final=False):
    # Carica il dataset di test
    if final == True:
        test_dataset = MoleculeDataset(root="data/test_data/TCMBANK", filename=test_data_path, test=True,final=final)
    else:
        test_dataset = MoleculeDataset(root="data/test_data", filename=test_data_path, test=True,final=final)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=8,pin_memory=True)

    # Definisci la loss function
    weight = torch.tensor([pos_weight], dtype=torch.float32).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    
    # Esegui il testing
    model.eval()
    all_preds = []
    all_labels = []
    all_smiles = []
    running_loss = 0.0
    step = 0
    high_attention_frags = []
    with torch.no_grad():
        prev_batch_num_graphs = 0
        for batch_index, (batch, tree_batch) in enumerate(tqdm(test_loader)):  #here train_loader is a DataLoader version specific for the graph data 
            batch.to(device)  
            tree_batch.to(device)
            data = {
            "mol_raws": {
                "x": batch.x.float(),
                "edge_attr": batch.edge_attr.float(),
                "edge_index": batch.edge_index,
                "batch": batch.batch,
            },
            "mol_trees": {
                "tree_node_features": tree_batch.x.float(),
                "tree_edge_attr":tree_batch.edge_attr.float(),
                "tree_edge_index": tree_batch.edge_index ,
                "tree_batch_index": tree_batch.batch,
            }
        }
            trees = tree_batch.mol_tree
            pred,raw_edge_index,raw_attention_scores,tree_edge_index,tree_attention_scores = model(data,device)

            num_graphs,high_score_fragments = highlight_mol_per_graph(raw_edge_index, raw_attention_scores, tree_edge_index,tree_attention_scores, batch,tree_batch,prev_batch_num_graphs,deep_search=deep_search,final=final)
            
            prev_batch_num_graphs += num_graphs
            high_attention_frags.extend(high_score_fragments)
            
            if final == False:
                loss = loss_fn(torch.squeeze(pred), batch.y.float())
                running_loss += loss.item()
                all_labels.append(batch.y.cpu().detach().numpy())

            step += 1
            all_preds.append(torch.sigmoid(pred).cpu().detach().numpy())
            # salva SMILES e predizione per ogni molecola del batch
            for i in range(len(batch.smiles)):
                all_smiles.append(batch.smiles[i])
            

            

    all_preds = np.concatenate(all_preds).ravel()
    if final == False:
        all_labels = np.concatenate(all_labels).ravel()
    print(f"Predictions: {all_preds}")

    if final == False:
        print(f"Labels: {all_labels}")
        test_loss = running_loss / step
    
    #calibrazione threshold
    

    # Calcola le metriche di valutazione
    if final == False:
        best_threshold = optimize_threshold(all_preds, all_labels)
        print(f"Best threshold: {best_threshold}")
        final_preds = (all_preds >= best_threshold).astype(int)
        calculate_metrics(final_preds, all_labels, 0, "test")
        log_conf_matrix(final_preds, all_labels, 0,final=final)
    else:
        final_preds = (all_preds >= 0.5).astype(int)
        file_name = "TCMBANK_molecule_predictions.csv"
        predictions_df = pd.DataFrame({"smiles": all_smiles, "pred": final_preds})
        predictions_df.to_csv(f"data/test_data/TCMBANK/{file_name}", index=False)
    
        



    # Salva i frammenti con attenzione più alta
    if final == False:
        high_attention_frags = [
        (smiles, num_neigh, id_molecola, final_preds[id_molecola],all_labels[id_molecola])
        for smiles, num_neigh, id_molecola in high_attention_frags
]
        high_attention_frags_df = pd.DataFrame(high_attention_frags, columns=["smiles","num_neigh","id_molecola","pred","real_label"])
    else:
        high_attention_frags = [
        (smiles, num_neigh, id_molecola, final_preds[id_molecola])
        for smiles, num_neigh, id_molecola in high_attention_frags
]
        high_attention_frags_df = pd.DataFrame(high_attention_frags, columns=["smiles","num_neigh","id_molecola","pred"])
        

    

    if deep_search == True:
        file_name = "deep_search_high_attention_frags.csv"
    else:
        file_name = "high_attention_frags.csv"

    if final == True:
        high_attention_frags_df.to_csv(f"data/test_data/TCMBANK/{file_name}", index=False)
    else:
        high_attention_frags_df.to_csv(f"data/test_data/{file_name}", index=False)

    # Aggiungi una colonna con le immagini molecolari
    # high_attention_frags_df['image'] = high_attention_frags_df['smiles'].apply(lambda x: Draw.MolToImage(Chem.MolFromSmiles(x)))
    

    #Analisi frammenti

    if final == False:
        print(f"Test Loss: {test_loss}")
        return test_loss
    else:
        return None


def load_config(model_name):
    model_name = model_name.split("\\")[2]
    print(f"Model name: {model_name}")
    model_name = model_name.split("_")[0]
    model_path = os.path.join("mlruns", "0")
    run_folders = os.listdir(model_path)
    matching_folder = None

    for folder in run_folders:
        if folder.startswith(model_name):
            matching_folder = folder
            break

    if matching_folder is not None:
        params_folder = os.path.join(model_path, matching_folder, "params")
        params = {}
        for file_name in os.listdir(params_folder):
            file_path = os.path.join(params_folder, file_name)
            with open(file_path, "r") as f:
                param_name = file_name.split(".")[0]
                param_value = f.read()
                params[param_name] = param_value
    
        return params
    else:
        assert matching_folder is not None, "Matching folder not found."
        return None

    
if __name__ == "__main__":
    
    #Aggiungi argomenti da linea di comando
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--deep_search", type=bool, required=True)
    # parser.add_argument("--final", type=bool, required=True)
    # args = parser.parse_args()
    deep_search = False
    final =True
    print(f"Deep search: {deep_search}")
    print(f"Final: {final}")


    # Determina il feature_size (facoltativo, se necessario)
    # train_dataset = MoleculeDataset(root="data", filename=params["train_file"]) 
    # feature_size = train_dataset[0].x.shape[1]
    feature_size = 30
    checkpoint_path = 'mlartifacts\\0\\388693edcfd747f6ae7f2edb65d65cd2\\artifacts\model\data\model.pth'
    params = load_config(checkpoint_path)
    print(f"Loaded params: {params}")

    # Carica il modello salvato come istanza
    model = load_model(checkpoint_path)

    # Esegui il testing
    if final == True:
        test_data_path = 'preprocessed_TMMC_compounds.csv'
        folder_path = 'data/test_data/TCMBANK/processed'
    else:
        test_data_path = 'new2_ZINC_final_balanced_test_set.csv'
        folder_path = 'data/test_data/processed'

    # Elimina i file nella cartella processed
    # for filename in os.listdir(folder_path):
    #     file_path = os.path.join(folder_path, filename)
    #     if os.path.isfile(file_path):
    #         print(f"Deleting file: {file_path}")
    #         os.remove(file_path)

    batch_size = int(params['batch_size'])
    pos_weight = float(params['pos_weight'])
    test_loss = test_model(model, test_data_path, batch_size, pos_weight,deep_search=deep_search,final=final)

    if final == False:
        print(f"Finished testing with test loss: {test_loss}")
    
