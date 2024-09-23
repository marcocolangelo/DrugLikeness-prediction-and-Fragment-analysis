#%% imports 
import os
import time
import torch 
from torch_geometric.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, \
    accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np
from tqdm import tqdm
from dataset_featurizer import MoleculeDataset
from model import GNN
import mlflow.pytorch
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 
from my_code.tree_building import DGLMolTree
from my_code.tree_path import myModel
import warnings
import logging
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("jax").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


warnings.filterwarnings("ignore")

import csv
import os
def gpu_check(model, data, interval=10):
    """
    Esegue i controlli sull'uso della GPU prima e durante l'addestramento.

    Args:
        model: Il modello PyTorch da addestrare.
        data: I dati utilizzati per l'addestramento, che devono essere sulla GPU.
        interval: L'intervallo di tempo (in secondi) tra un controllo e l'altro durante l'addestramento.
    """

    # Controllo se la GPU è disponibile
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Sposta il modello sul dispositivo
    model.to(device)
    
    # # Sposta i dati sul dispositivo
    # data = data.to(device)

    # # Controllo se il modello è sulla GPU
    # model_device = next(model.parameters()).device
    # print(f"Model is on: {model_device}")

    # # Controllo se uno dei tensori dei dati è sulla GPU
    # data_device = next(data[key].device for key in data.keys() if isinstance(data[key], torch.Tensor))
    # print(f"Data is on: {data_device}")

    # # Verifica se modello e dati sono effettivamente sulla GPU
    # assert model_device.type == 'cuda', "Model is not on GPU!"
    # assert data_device.type == 'cuda', "Data is not on GPU!"

    # # Funzione per monitorare l'uso della GPU durante l'addestramento
    def monitor_gpu_usage():
        while True:
            print(f"Memory allocated: {torch.cuda.memory_allocated()} bytes")
            print(f"Memory reserved: {torch.cuda.memory_reserved()} bytes")
            time.sleep(interval)

    # # Avvia il monitoraggio dell'uso della GPU in un thread separato
    import threading
    monitor_thread = threading.Thread(target=monitor_gpu_usage)
    monitor_thread.daemon = True
    monitor_thread.start()

    return device


def get_next_run_id(file_path):
    if not os.path.isfile(file_path):
        return 0
    
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader, None)  # Salta l'header
        ids = [int(row[0]) for row in reader if row]
    
    if not ids:
        return 0
    return max(ids) + 1


run_id = get_next_run_id('model_results/training_results.csv')

def save_results(file_path, params, best_train_loss, best_test_loss):
    
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Scrivi l'header solo se il file non esiste
            header = ["run_id"]+ list(params.keys()) + ['best_train_loss', 'best_test_loss']
            writer.writerow(header)
        row = [run_id]+ list(params.values()) + [best_train_loss, best_test_loss]
        writer.writerow(row)





def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(epoch, model, train_loader, optimizer, loss_fn):
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
        pred, _, _,_,_ = model(data, device)
        

        # Calculating the loss and gradients
        loss = loss_fn(torch.squeeze(pred), batch.y.float())
        loss.backward()  
        optimizer.step()  
        # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        all_labels.append(batch.y.cpu().detach().numpy())
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels, epoch, "train")
    return running_loss/step

def test(epoch, model, test_loader, loss_fn):
    all_preds = []
    all_preds_raw = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for batch_index, (batch, tree_batch) in enumerate(tqdm(test_loader)):
        # Ora hai accesso a `batch` per le molecole e `tree_batch` per gli alberi.

        batch.to(device)  
        tree_batch.to(device)
        # Create a mapping from global indices to local batch indices
        #NON VERIFICATO SE VA BENE O NO!!!
        # unique_molecule_ids = batch.tree_batch_index.unique(sorted=True)
        # id_mapping = {global_id.item(): local_id for local_id, global_id in enumerate(unique_molecule_ids)}

        # # Trasferisci il tensore sulla CPU se non lo è già
        # tree_batch_index_cpu = batch.tree_batch_index.cpu()

        # # Applica la mappatura sulla CPU
        # tree_batch_index_cpu.apply_(lambda x: id_mapping[x])

        # # Se necessario, riporta il tensore al dispositivo originale
        # batch.tree_batch_index = tree_batch_index_cpu.to(device)
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
                "tree_edge_attr": tree_batch.edge_attr.float(),
                "tree_edge_index": tree_batch.edge_index,
                "tree_batch_index": tree_batch.batch,
            }
        }

        # print(f"mol_trees: {data['mol_trees']['tree_node_features'].size()}")
        # print(f"mol_trees: {data['mol_trees']['tree_edge_attr'].size()}")
        # print(f"mol_trees: {data['mol_trees']['tree_edge_index'].size()}")
        # print(f"mol_trees: {data['mol_trees']['tree_batch_index'].size()}")


        # Forward pass
        pred, _, _,_,_ = model(data, device)
        loss = loss_fn(torch.squeeze(pred), batch.y.float())

         # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        all_preds_raw.append(torch.sigmoid(pred).cpu().detach().numpy())
        all_labels.append(batch.y.cpu().detach().numpy())
    
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    # print(all_preds_raw[0][:10])
    # print(all_preds[:10])
    # print(all_labels[:10])
    calculate_metrics(all_preds, all_labels, epoch, "test")
    log_conf_matrix(all_preds, all_labels, epoch)
    return running_loss/step

def log_conf_matrix(y_pred, y_true, epoch):
    # Log confusion matrix as image
    cm = confusion_matrix(y_pred, y_true)
    classes = ["0", "1"]
    df_cfm = pd.DataFrame(cm, index = classes, columns = classes)
    plt.figure(figsize = (10,7))
    cfm_plot = sns.heatmap(df_cfm, annot=True, cmap='Blues', fmt='g')
    cfm_plot.figure.savefig(f'data/images/cm_{epoch}.png')
    mlflow.log_artifact(f"data/images/cm_{epoch}.png")

def calculate_metrics(y_pred, y_true, epoch, type):
    print(f"\n Confusion matrix: \n {confusion_matrix(y_pred, y_true)}")
    print(f"F1 Score: {f1_score(y_true, y_pred)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    mlflow.log_metric(key=f"Precision-{type}", value=float(prec), step=epoch)
    mlflow.log_metric(key=f"Recall-{type}", value=float(rec), step=epoch)
    try:
        roc = roc_auc_score(y_true, y_pred)
        print(f"ROC AUC: {roc}")
        mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(roc), step=epoch)
    except:
        mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(0), step=epoch)
        print(f"ROC AUC: notdefined")

# from torch_geometric.data import Batch

# class CustomBatch(Batch):
#     @staticmethod
#     def from_data_list(data_list):
#         batch = CustomBatch()
#         for key in data_list[0].keys:
#             batch[key] = []
#         for data in data_list:
#             for key in data.keys:
#                 batch[key].append(data[key])

#         # Qui unisci i tuoi attributi personalizzati
#         if 'tree_batch_index' in data_list[0]:
#             print("tree_batch_index")
#             batch.tree_batch_index = torch.cat([data.tree_batch_index for data in data_list], dim=0)
        
#         return batch.contiguous()
    
# class CustomCollater(object):
#     def __init__(self, follow_batch, exclude_keys):
#         self.follow_batch = follow_batch
#         self.exclude_keys = exclude_keys

#     def collate(self, batch):
#         return CustomBatch.from_data_list(batch)

#     def __call__(self, batch):
#         return self.collate(batch)

#%% Run the training
from mango import Tuner
from config import HYPERPARAMETERS, BEST_PARAMETERS, SIGNATURE

def run_one_training(params):
    params = params[0]
    train_losses = []
    test_losses = []
    
    with mlflow.start_run() as run:
        # Log parameters used in this experiment
        for key in params.keys():
            mlflow.log_param(key, params[key])

        # Loading the dataset
        print("Loading dataset...")
        train_dataset = MoleculeDataset(root="data/", filename=params["train_file"] )
        test_dataset = MoleculeDataset(root="data/", filename=params["test_file"], test=True)
        # tree_train_dataset = TreeDataset(root="data/", filename=params["train_file"] )
        # tree_test_dataset = TreeDataset(root="data/", filename=params["test_file"], test=True)

        params["model_edge_dim"] = train_dataset[0][0].edge_attr.shape[1] #train_dataset[0] is the first molecule in the dataset
        print(f"params: {params}")
        # Prepare training
        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True,num_workers=10,pin_memory=True) 
        test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=True,num_workers=5,pin_memory=True)

        # tree_train_loader = DataLoader(tree_train_dataset, batch_size=params["batch_size"], shuffle=True)
        # tree_test_loader = DataLoader(tree_test_dataset, batch_size=params["batch_size"], shuffle=True)

        #stampa grandezza dataset
        # print(f"Train dataset size: {len(train_dataset)}")
        # print(f"Test dataset size: {len(test_dataset)}")
        # print(f"Train tree dataset size: {len(tree_train_dataset)}")
        # print(f"Test tree dataset size: {len(tree_test_dataset)}")

        # Loading the model
        print("Loading model...")
        model_params = {k: v for k, v in params.items() if k.startswith("model_")}
        # model = GNN(feature_size=train_dataset[0].x.shape[1], model_params=model_params)  #train_dataset[i].x.shape[1] is the number of features in the i-th molecule
        model = myModel(feature_size=train_dataset[0][0].x.shape[1], model_params=model_params)
        print(f"Model parameters: {model_params}")
        model = model.to(device)
        # device_check = gpu_check(model, next(iter(train_loader)))
        # print(f"Model on: {device_check}")
        print(f"Number of parameters: {count_parameters(model)}")
        mlflow.log_param("num_params", count_parameters(model))

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

                ######## questa era la versione originale ########
                # loss = train_one_epoch(epoch, model, train_loader, optimizer, loss_fn)

                # versione nuova che inserisce anche il tree_loader
                loss = train_one_epoch(epoch, model,train_loader, optimizer, loss_fn)

                train_losses.append(loss)
                print(f"Epoch {epoch} | Train Loss {loss}")
                mlflow.log_metric(key="Train loss", value=float(loss), step=epoch)

                # Testing
                model.eval()
                if epoch % 5 == 0:
                    ######## questa era la versione originale ########
                    # loss = test(epoch, model, test_loader, loss_fn)

                    # versione nuova che inserisce anche il tree_loader
                    loss = test(epoch, model, test_loader, loss_fn)
                    test_losses.append(loss)
                    print(f"Epoch {epoch} | Test Loss {loss}")
                    mlflow.log_metric(key="Test loss", value=float(loss), step=epoch)
                    
                    # Update best loss
                    if float(loss) < best_loss:
                        best_loss = loss
                        # Save the currently best model 
                        mlflow.pytorch.log_model(model, "model", signature=SIGNATURE)
                        print("Saving best model...")
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1

                scheduler.step()
            else:
                print("Early stopping due to no improvement.")
                plot_loss_curves(train_losses, test_losses)
                return [best_loss]
    plot_loss_curves(train_losses, test_losses)
    print(f"Finishing training with best test loss: {best_loss}")
    save_results('model_results/training_results.csv', params, min(train_losses), min(test_losses))
    return [best_loss]

def plot_loss_curves(train_losses, test_losses):
    # Plotting the loss curves
    # id = get_next_run_id('model_results/training_results.csv')
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(test_losses) * 5 + 1, 5), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Test Loss Curves')
    plt.savefig(f"data/images/loss_curves_config_{run_id}.png")
    mlflow.log_artifact(f"data/images/loss_curves_config_{run_id}.png")
    
    plt.show()
    






# # Create a dummy model
if __name__ == "__main__":
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Specify tracking server
    mlflow.set_tracking_uri("http://localhost:5000")
  

    # %% Hyperparameter search
    # print("Running hyperparameter search...")
    config = dict()
    # config["optimizer"] = "Bayesian"

    # # Questa config pare essere buona anche
    # params={'batch_size': 16, 'learning_rate': 0.0014529236861494325, 'weight_decay': 7.23152103258082e-05, 'sgd_momentum': 0.8946112505430376, 'scheduler_gamma': 0.8805917002592172, 'pos_weight': 1.3, 'model_embedding_size': 128, 'model_attention_heads': 4, 'model_layers': 1, 'model_dropout_rate': 0.2658137351601765, 'model_top_k_ratio': 0.745862388949835, 'model_top_k_every_n': 3, 'model_dense_neurons': 512, 'epochs': 20, 'train_file': 'medium_new2_ZINC_final_balanced_train_set.csv', 'test_file': 'small_new2_ZINC_final_balanced_validation_set.csv', 'model_edge_dim': 11}
# Model parameters: {'model_embedding_size': 128, 'model_attention_heads': 4, 'model_layers': 1, 'model_dropout_rate': 0.2658137351601765, 'model_top_k_ratio': 0.745862388949835, 'model_top_k_every_n': 3, 'model_dense_neurons': 512, 'model_edge_dim': 11}

    # BEST_PARAMETERS = {
    # 'batch_size': 64, #prima era 64!!!!!!!!!!
    # 'learning_rate': 0.014015474347793431, 
    # 'weight_decay': 0.002449693011254293, 
    # 'sgd_momentum': 0.8357800974603684, 
    # 'scheduler_gamma': 0.972996507445344, 
    # 'pos_weight': 1.0, 
    # 'model_embedding_size': 256, 
    # 'model_attention_heads': 2, 
    # 'model_layers': 2, 
    # 'model_dropout_rate': 0.20040422634492724, 
    # 'model_top_k_ratio': 0.5995492263146199, 
    # 'model_top_k_every_n': 1, 
    # 'model_dense_neurons': 128,
    #     'epochs': 201,
    #     'train_file': "new2_ZINC_final_balanced_train_set.csv",
    #     'test_file': "new2_ZINC_final_balanced_validation_set.csv"
    # }

    BEST_PARAMETERS = {
        'batch_size': 32, 
        'learning_rate': 0.0011862515762923046, 
        'weight_decay': 0.0005073072978716553, 
        'sgd_momentum': 0.8295220628523434, 
        'scheduler_gamma': 0.9059721848106833, 
        'pos_weight': 0.7, 
        'model_embedding_size': 256, 
        'model_attention_heads': 3, 
        'model_layers': 1,
        'model_dropout_rate': 0.22277792420399223, 
        'model_top_k_ratio': 0.28771088919741555, 
        'model_top_k_every_n': 1, 
        'model_dense_neurons': 256,
        'epochs': 201,
        'train_file': "new2_ZINC_final_balanced_train_set.csv",
        'test_file': "new2_ZINC_final_balanced_validation_set.csv"}

    
    

    # # Create config using BEST_PARAMETERS
    for key in BEST_PARAMETERS.keys():
        config[key] = BEST_PARAMETERS[key]


    running_loss = run_one_training([config])
    # tuner = Tuner(BEST_PARAMETERS, 
    #               objective=run_one_training,
    #               conf_dict=config) 
    # results = tuner.minimize()

