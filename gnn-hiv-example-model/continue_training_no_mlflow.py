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
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 
from config import HYPERPARAMETERS, BEST_PARAMETERS, SIGNATURE

import csv
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    

    # Funzione per monitorare l'uso della GPU durante l'addestramento
    def monitor_gpu_usage():
        while True:
            print(f"Memory allocated: {torch.cuda.memory_allocated()} bytes")
            print(f"Memory reserved: {torch.cuda.memory_reserved()} bytes")
            time.sleep(interval)

    # Avvia il monitoraggio dell'uso della GPU in un thread separato
    import threading
    monitor_thread = threading.Thread(target=monitor_gpu_usage)
    monitor_thread.daemon = True
    monitor_thread.start()

    return device

def load_pretrained_model(checkpoint_dir, feature_size, model_params):
    model_path = os.path.join(checkpoint_dir, 'model.pth')
    print(f"Loading model from {model_path}")
    model = GNN(**model_params)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    print(f"Model loaded correctly")
    # Verifica che il modello sia stato caricato correttamente sul device
    print(f"Device del modello: {next(model.parameters()).device}")
    return model

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

def save_results(file_path, run_id, params, best_train_loss, best_test_loss):
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Scrivi l'header solo se il file non esiste
            header = ["run_id"] + list(params.keys()) + ['best_train_loss', 'best_test_loss']
            writer.writerow(header)
        row = [run_id] + list(params.values()) + [best_train_loss, best_test_loss]
        writer.writerow(row)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one_epoch(epoch, model, train_loader, optimizer, loss_fn, checkpoint_dir):
    all_preds = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for batch_index, (batch, tree_batch) in enumerate(tqdm(train_loader)):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch.to(device, non_blocking=True)  
        tree_batch.to(device, non_blocking=True)
        optimizer.zero_grad() 
        
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

        pred, _, _, _, _ = model(data, device)
        
        loss = loss_fn(torch.squeeze(pred), batch.y.float())
        loss.backward()  
        optimizer.step()  
        running_loss += loss.item()
        step += 1
        all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        all_labels.append(batch.y.cpu().detach().numpy())
    
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    metrics = calculate_metrics(all_preds, all_labels, epoch, "train", checkpoint_dir)
    return running_loss / step

def test(epoch, model, test_loader, loss_fn, checkpoint_dir):
    all_preds = []
    all_preds_raw = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for batch_index, (batch, tree_batch) in enumerate(test_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        pred, _, _, _, _ = model(data, device)
        loss = loss_fn(torch.squeeze(pred), batch.y.float())
        running_loss += loss.item()
        step += 1
        all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        all_preds_raw.append(torch.sigmoid(pred).cpu().detach().numpy())
        all_labels.append(batch.y.cpu().detach().numpy())
    
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    print(all_preds_raw[0][:10])
    print(all_preds[:10])
    print(all_labels[:10])
    metrics = calculate_metrics(all_preds, all_labels, epoch, "test", checkpoint_dir)
    log_conf_matrix(all_preds, all_labels, epoch, checkpoint_dir)
    return running_loss / step

def log_conf_matrix(y_pred, y_true, epoch, checkpoint_dir):
    cm = confusion_matrix(y_true, y_pred)
    classes = ["0", "1"]
    df_cfm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(10, 7))
    cfm_plot = sns.heatmap(df_cfm, annot=True, cmap='Blues', fmt='g')
    image_path = os.path.join(checkpoint_dir, f'cm_{epoch}.png')
    cfm_plot.figure.savefig(image_path)
    plt.close()
    print(f"Confusion matrix saved at {image_path}")

def calculate_metrics(y_pred, y_true, epoch, data_type, checkpoint_dir):
    print(f"\n Confusion matrix: \n {confusion_matrix(y_true, y_pred)}")
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    try:
        roc = roc_auc_score(y_true, y_pred)
    except:
        roc = 0
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print(f"ROC AUC: {roc}")

    metrics = {
        f"Precision-{data_type}": prec,
        f"Recall-{data_type}": rec,
        f"ROC-AUC-{data_type}": roc,
        f"F1-{data_type}": f1,
        f"Accuracy-{data_type}": accuracy,
        "epoch": epoch
    }
    log_metrics(metrics, checkpoint_dir)
    return metrics

def log_metrics(metrics, checkpoint_dir):
    file_path = os.path.join(checkpoint_dir, 'metrics.csv')
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as csvfile:
        fieldnames = metrics.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)

def save_params(params, checkpoint_dir):
    file_path = os.path.join(checkpoint_dir, 'params.json')
    with open(file_path, 'w') as f:
        json.dump(params, f, indent=4)

def continue_training(params, checkpoint_dir, continue_epochs, run_id):
    params = params[0]
    train_losses = []
    test_losses = []

    # Ensure the checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save parameters
    save_params(params, checkpoint_dir)

    print("Loading dataset...")
    train_dataset = MoleculeDataset(root="data/", filename=params["train_file"])
    test_dataset = MoleculeDataset(root="data/", filename=params["test_file"], test=True)
    params["model_edge_dim"] = train_dataset[0][0].edge_attr.shape[1]
    print(f"params: {params}")

    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=6, pin_memory=True)
    print(f"Train dataset size: {len(train_dataset)}")

    print("Loading pretrained model...")
    model_params = {k.replace("model_", ""): v for k, v in params.items() if k.startswith("model_")}
    feature_size = train_dataset[0][0].x.shape[1]
    model = load_pretrained_model(checkpoint_dir, feature_size, model_params)
    print(f"Number of parameters: {count_parameters(model)}")

    weight = torch.tensor([params["pos_weight"]], dtype=torch.float32).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=params["learning_rate"],
                                momentum=params["sgd_momentum"],
                                weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params["scheduler_gamma"])

    best_loss = 1000
    early_stopping_counter = 0
    for epoch in range(continue_epochs):
        if early_stopping_counter <= 20:
            model.train()
            loss = train_one_epoch(epoch, model, train_loader, optimizer, loss_fn, checkpoint_dir)
            train_losses.append(loss)
            print(f"Epoch {epoch} | Train Loss {loss}")

            # Log training loss
            log_metrics({'Train loss': loss, 'epoch': epoch}, checkpoint_dir)

            model.eval()
            if epoch % 5 == 0:
                loss = test(epoch, model, test_loader, loss_fn, checkpoint_dir)
                test_losses.append(loss)
                print(f"Epoch {epoch} | Test Loss {loss}")

                # Log test loss
                log_metrics({'Test loss': loss, 'epoch': epoch}, checkpoint_dir)

                plot_loss_curves(train_losses, test_losses, run_id, checkpoint_dir)
                
                if float(loss) < best_loss:
                    best_loss = loss
                    # Save the model
                    model_save_path = os.path.join(checkpoint_dir, 'model.pth')
                    torch.save(model.state_dict(), model_save_path)
                    print(f"Model saved at {model_save_path}")
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

            scheduler.step()
        else:
            print("Early stopping due to no improvement.")
            plot_loss_curves(train_losses, test_losses, run_id, checkpoint_dir)
            save_results(os.path.join(checkpoint_dir, 'training_results.csv'), run_id, params, min(train_losses), min(test_losses))
            return [best_loss]
    plot_loss_curves(train_losses, test_losses, run_id, checkpoint_dir)
    print(f"Finishing training with best test loss: {best_loss}")
    save_results(os.path.join(checkpoint_dir, 'training_results.csv'), run_id, params, min(train_losses), min(test_losses))
    return [best_loss]

def plot_loss_curves(train_losses, test_losses, run_id, checkpoint_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(test_losses) * 5 + 1, 5), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Test Loss Curves')
    image_path = os.path.join(checkpoint_dir, f"loss_curves_config_{run_id}.png")
    plt.savefig(image_path)
    plt.show()
    print(f"Loss curves saved at {image_path}")

if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # %% Continue Training
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
        'epochs': 500,
        'train_file': "new2_ZINC_final_balanced_train_set.csv",
        'test_file': "new2_ZINC_final_balanced_validation_set.csv"
    }

    config = {key: BEST_PARAMETERS[key] for key in BEST_PARAMETERS.keys()}

    # Set the checkpoint directory where all materials will be saved
    checkpoint_dir = 'models\\batch32_weight0-7_big_datasets'

    # Ensure the checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Get the run ID
    run_id = get_next_run_id(os.path.join(checkpoint_dir, 'training_results.csv'))

    continue_epochs = 500  # Number of additional epochs to continue training

    running_loss = continue_training([config], checkpoint_dir=checkpoint_dir, continue_epochs=continue_epochs, run_id=run_id)
