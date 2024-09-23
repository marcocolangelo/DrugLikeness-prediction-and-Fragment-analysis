# %% imports
import csv
import io
import logging
import os
import sys
import warnings
import torch
from torch_geometric.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np
from tqdm import tqdm
from dataset_featurizer import MoleculeDataset
from model import GNN
# import mlflow.pytorch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
import optuna
from config import HYPERPARAMETERS, BEST_PARAMETERS, SIGNATURE
from my_code.tree_path import myModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.getLogger("deepchem").setLevel(logging.ERROR)
logging.getLogger("deep-chem").setLevel(logging.ERROR)
logging.getLogger("deep_chem").setLevel(logging.ERROR)




warnings.filterwarnings("ignore")
# Specify tracking server
# mlflow.set_tracking_uri("http://localhost:5000")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# def validate(epoch, model, val_loader, loss_fn):
#     all_preds = []
#     all_labels = []
#     running_loss = 0.0
#     step = 0
#     model.eval()
#     with torch.no_grad():
#         for batch in val_loader:
#             batch.to(device)
#             pred = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch)
#             loss = loss_fn(torch.squeeze(pred), batch.y.float())
#             running_loss += loss.item()
#             step += 1
#             all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
#             all_labels.append(batch.y.cpu().detach().numpy())
#     all_preds = np.concatenate(all_preds).ravel()
#     all_labels = np.concatenate(all_labels).ravel()
#     calculate_metrics(all_preds, all_labels, epoch, "validation")
#     return running_loss / step

def train_one_epoch(epoch, model, train_loader, optimizer, loss_fn,trial = None):
    # Enumerate over the data
    all_preds = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for batch_index, (batch, tree_batch) in enumerate(tqdm(train_loader)):  #here train_loader is a DataLoader version specific for the graph data 
        #batch is the data for the current batch in the DataLoader
        # Use GPU
        batch.to(device)  
        tree_batch.to(device)
        # Reset gradients
        optimizer.zero_grad() 
        # Estrazione delle features dalle molecole
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

      
        # Forward pass
        pred,_,_,_,_ = model(data, device)
        

        # Calculating the loss and gradients
        try:
            loss = loss_fn(torch.squeeze(pred), batch.y.float())
        except:
            # print(f"Error in loss calculation at epoch {epoch} and step {step}")
            # print(f"Pred shape: {pred.shape}")
            # print(f"Batch shape: {batch.y.shape}")
            # print(f"Batch: {batch.y}")
            # print(f"Pred: {pred}")
            
            loss = loss_fn(pred, batch.y.view(-1, 1).float())

        loss.backward()  
        optimizer.step()  
        # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        all_labels.append(batch.y.cpu().detach().numpy())
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    f1,acc,prec,rec,roc  = calculate_metrics(all_preds, all_labels)

    # # Pruning basato solo sulla loss
    # if trial is not None:
    #     trial.report(running_loss / step, epoch)
    #     if trial.should_prune():
    #         raise optuna.exceptions.TrialPruned()

    return running_loss / step


def test(epoch, model, test_loader, loss_fn,prova_id):
    all_preds = []
    all_preds_raw = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for batch_index, (batch, tree_batch) in enumerate(tqdm(test_loader)):

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
                "tree_edge_attr": tree_batch.edge_attr.float(),
                "tree_edge_index": tree_batch.edge_index,
                "tree_batch_index": tree_batch.batch,
            }
        }


        # Forward pass
        pred,_,_ = model(data, device)
        loss = loss_fn(torch.squeeze(pred), batch.y.float())

         # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        all_preds_raw.append(torch.sigmoid(pred).cpu().detach().numpy())
        all_labels.append(batch.y.cpu().detach().numpy())
    
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    f1,acc,prec,rec,roc  = calculate_metrics(all_preds, all_labels)
    log_conf_matrix(all_preds, all_labels, epoch,provaid=prova_id)
    return running_loss/step, f1,acc,prec,rec,roc

def log_conf_matrix(y_pred, y_true, epoch,provaid):
    cm = confusion_matrix(y_true, y_pred)
    classes = ["0", "1"]
    df_cfm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(10, 7))
    cfm_plot = sns.heatmap(df_cfm, annot=True, cmap='Blues', fmt='g')

    #crea cartella se non esiste
    os.makedirs(f'data/validation/images/conf_matrix/prova_{provaid}', exist_ok=True)
    cfm_plot.figure.savefig(f'data/validation/images/conf_matrix/prova_{provaid}/cm_{epoch}.png')
    # mlflow.log_artifact(f"data/images/cm_{epoch}.png")

def calculate_metrics(y_pred, y_true):
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    print(f"\n Confusion matrix: \n {confusion_matrix(y_true, y_pred)}")
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {acc}")
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    # mlflow.log_metric(key=f"Precision-{split}", value=float(prec), step=epoch)
    # mlflow.log_metric(key=f"Recall-{split}", value=float(rec), step=epoch)
    try:
        roc = roc_auc_score(y_true, y_pred)
        print(f"ROC AUC: {roc}")
        # mlflow.log_metric(key=f"ROC-AUC-{split}", value=float(roc), step=epoch)
    except:
        # mlflow.log_metric(key=f"ROC-AUC-{split}", value=float(0), step=epoch)
        roc = 0
        print(f"ROC AUC: not defined")
    return f1,acc,prec,rec,roc

# %% Funzione per il training
def run_one_training(params):
    params = params[0]
    train_losses = []
    test_losses = []
    id_prova = params["id_prova"]
    trial = params["trial_obj"]
    # with mlflow.start_run() as run:
        # Log parameters used in this experiment
        # for key in params.keys():
        #     mlflow.log_param(key, params[key])

    # Loading the dataset
    print("Loading dataset...")
    train_dataset = MoleculeDataset(root="data/validation", filename=params["train_file"] )
    test_dataset = MoleculeDataset(root="data/validation", filename=params["test_file"], test=True)

    # Update the model parameters
    params["model_edge_dim"] = train_dataset[0][0].edge_attr.shape[1] #train_dataset[0] is the first molecule in the dataset
    print(f"params: {params}")

    # Prepare training
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True,num_workers=6,pin_memory=True) 
    test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=True,num_workers=6,pin_memory=True)
 
    # Loading the model
    print("Loading model...")
    model_params = {k: v for k, v in params.items() if k.startswith("model_")}
    model = myModel(feature_size=train_dataset[0][0].x.shape[1], model_params=model_params)
    print(f"Model parameters: {model_params}")
    model = model.to(device)
    # device_check = gpu_check(model, next(iter(train_loader)))
    # print(f"Model on: {device_check}")
    print(f"Number of parameters: {count_parameters(model)}")
    # mlflow.log_param("num_params", count_parameters(model))

    # < 1 increases precision, > 1 recall
    weight = torch.tensor([params["pos_weight"]], dtype=torch.float32).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=params["learning_rate"],
                                momentum=params["sgd_momentum"],
                                weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params["scheduler_gamma"])

    # Start training
    best_loss = 1000
    early_stopping_counter = 0

    for epoch in tqdm(range(params["epochs"])):
        if early_stopping_counter <= 25: # = x * 5 
            # Training
            model.train()

            
            # versione nuova che inserisce anche il tree_loader
            loss = train_one_epoch(epoch, model,train_loader, optimizer, loss_fn,trial)

            train_losses.append(loss)
            print(f"Epoch {epoch} | Train Loss {loss}")
            # mlflow.log_metric(key="Train loss", value=float(loss), step=epoch)

            # Testing
            model.eval()
            if epoch % 5 == 0:
                # versione nuova che inserisce anche il tree_loader
                loss,f1,acc,prec,rec,roc = test(epoch, model, test_loader, loss_fn,prova_id=id_prova)
                test_losses.append(loss)
                print(f"Epoch {epoch} | Test Loss {loss}")
                # mlflow.log_metric(key="Test loss", value=float(loss), step=epoch)
                
                # Update best loss
                if float(loss) < best_loss:
                    best_loss = loss
                    best_metrics = {"f1": f1, "acc": acc, "prec": prec, "rec": rec, "roc": roc}
                    # Save the currently best model 
                    # mlflow.pytorch.log_model(model, "model", signature=SIGNATURE)
                    print("Saving best model...")
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

            scheduler.step()
        else:
            print("Early stopping due to no improvement.")
            plot_loss_curves(train_losses, test_losses,id_prova)
            return best_loss,best_metrics
    plot_loss_curves(train_losses, test_losses,id_prova)
    print(f"Finishing training with best test loss: {best_loss} and best metrics: {best_metrics}")
    # save_results('model_results/training_results.csv', params, min(train_losses), min(test_losses))
    return best_loss,best_metrics

def plot_loss_curves(train_losses, val_losses,id_prova):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) * 5 + 1, 5), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Curves')
    plt.savefig(f"data/validation/images/losses/loss_curves_prova{id_prova}.png")


# %% Ottimizzazione con Optuna
# Funzione per salvare i risultati su un file CSV
def save_trial_results(trial_number, params, loss, metrics, filename="data/validation/optuna_results.csv"):
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Scrivi l'intestazione solo se il file non esiste
            header = ["trial_number"] + list(params.keys()) + ["loss"] + list(metrics.keys())
            writer.writerow(header)
        
        # Scrivi i risultati della prova
        row = [trial_number] + list(params.values()) + [loss] + list(metrics.values())
        writer.writerow(row)



def objective(trial):
    # Definisci i parametri utilizzando il trial corrente
    params = {
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.0001, 0.1),
        "weight_decay": trial.suggest_loguniform("weight_decay", 0.00001, 0.01),
        "sgd_momentum": trial.suggest_uniform("sgd_momentum", 0.8, 0.99),
        "scheduler_gamma": trial.suggest_uniform("scheduler_gamma", 0.8, 0.99),
        "pos_weight": trial.suggest_categorical("pos_weight", [0.5, 0.7, 0.9, 1.0, 1.3, 1.5]),
        "model_embedding_size": trial.suggest_categorical("model_embedding_size", [64, 128, 256, 512]),
        "model_attention_heads": trial.suggest_categorical("model_attention_heads", [1, 2, 3, 4, 5]),
        "model_layers": trial.suggest_categorical("model_layers", [1, 2, 3, 4, 5]),
        "model_dropout_rate": trial.suggest_uniform("model_dropout_rate", 0.2, 0.8),
        "model_top_k_ratio": trial.suggest_uniform("model_top_k_ratio", 0.25, 0.8),
        "model_top_k_every_n": trial.suggest_categorical("model_top_k_every_n", [1, 2, 3, 5]),
        "model_dense_neurons": trial.suggest_categorical("model_dense_neurons", [128, 256, 512]),
        'epochs': 21,
        'train_file': "medium_new2_ZINC_final_balanced_train_set.csv",
        'test_file': "new2_ZINC_final_balanced_validation_set.csv"
    }

    # Verifica se i parametri attuali sono già stati testati
    for past_trial in study.trials:
        if set(past_trial.params.items()) == set(params.items()):  # Confronta direttamente i parametri proposti dal trial corrente
            print(f"Configurazione già testata, salto questo trial: {params}")
            return None  # Salta questo trial
        
    # print(f"Testing configuration: {params}")
    params["id_prova"] = trial.number
    params["trial_obj"] = trial
    loss, metrics = run_one_training([params])

    roc = metrics["roc"]

    # Salva i risultati della prova in un file CSV
    save_trial_results(trial.number, params, loss, metrics)
    return loss, -roc

distributions = {
    "batch_size": optuna.distributions.CategoricalDistribution([16, 32, 64, 128]),
    "learning_rate": optuna.distributions.LogUniformDistribution(0.0001, 0.1),
    "weight_decay": optuna.distributions.LogUniformDistribution(0.00001, 0.01),
    "sgd_momentum": optuna.distributions.UniformDistribution(0.8, 0.99),
    "scheduler_gamma": optuna.distributions.UniformDistribution(0.8, 0.99),
    "pos_weight": optuna.distributions.CategoricalDistribution([0.5, 0.7, 0.9, 1.0, 1.3, 1.5]),
    "model_embedding_size": optuna.distributions.CategoricalDistribution([64, 128, 256, 512]),
    "model_attention_heads": optuna.distributions.CategoricalDistribution([1, 2, 3, 4, 5]),
    "model_layers": optuna.distributions.CategoricalDistribution([1, 2, 3, 4, 5]),
    "model_dropout_rate": optuna.distributions.UniformDistribution(0.2, 0.8),
    "model_top_k_ratio": optuna.distributions.UniformDistribution(0.25, 0.8),
    "model_top_k_every_n": optuna.distributions.CategoricalDistribution([1, 2, 3, 5]),
    "model_dense_neurons": optuna.distributions.CategoricalDistribution([128, 256, 512]),
}


if __name__ == "__main__":
    df = pd.read_csv('data/validation/optuna_results.csv')

    # Assicurati che le colonne siano numeriche
    # df['trial_obj'] = pd.to_numeric(df['trial_obj'], errors='coerce')
    # df['roc'] = pd.to_numeric(df['roc'], errors='coerce')

    # Filtra le righe con valori NaN nelle metriche
    # df = df.dropna(subset=['trial_obj', 'roc'])
    load_state = False
    n_trials = 150

    study = optuna.create_study(directions=["minimize", "minimize"], load_if_exists=load_state)
    
    if load_state == True:
        # Ottieni il numero di trial attuali nello studio
        existing_trials = len(study.trials)
        print(f"Numero di trial esistenti nello studio: {existing_trials}")

        # Aggiungi i risultati precedenti allo studio
        for _, row in df.iterrows():
            trial = optuna.trial.create_trial(
                params={key: row[key] for key in distributions.keys()},  # Parametri del trial
                distributions=distributions,  # Distribuzioni dei parametri
                values=[row['loss'],row["roc"]]   # Valore dell'obiettivo
            )
            study.add_trial(trial)

        # Verifica se tutti i trial sono stati aggiunti
        total_trials = len(study.trials)
        print(f"Numero di trial nel CSV: {len(df)}")
        print(f"Numero totale di trial dopo l'aggiunta: {total_trials}")
        
        if total_trials - existing_trials == len(df):
            print("Tutti i trial passati sono stati aggiunti correttamente.")
        else:
            print(f"Attenzione: Non tutti i trial sono stati aggiunti correttamente. Aggiunti: {total_trials - existing_trials} su {len(df)}")


    study.optimize(objective, n_trials=n_trials,show_progress_bar=True)

     # Migliore trial basato su entrambe le metriche
    best_trial = study.best_trials[0]  # Questo assume che il primo sia il migliore. Puoi iterare per trovare il migliore secondo un criterio specifico.

    print("Best trial:")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Stampa i parametri del miglior trial
    print("Best hyperparameters:", best_trial.params)
    print("Best loss:", best_trial.values[0])
    print("Best ROC AUC:", -best_trial.values[1])
