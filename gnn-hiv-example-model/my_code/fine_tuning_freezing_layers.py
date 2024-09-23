#%% imports 
import os
import csv
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Specify tracking server
mlflow.set_tracking_uri("http://localhost:5000")


# %% Utility functions  (fine_tuning_freezing_layers.py)
# Serve per congelare i primi n layer del modello
def freeze_layers(model, num_conv_layers_to_freeze):
    layer_count = 0
    for name, param in model.named_parameters():
        if "conv1" in name or (name.startswith("conv_layers") and layer_count < num_conv_layers_to_freeze):
            param.requires_grad = False
        else:
            param.requires_grad = True
        if name.startswith("conv_layers"):
            layer_count += 1

# Serve per scongelare i primi n layer del modello
def unfreeze_layers(model, num_conv_layers_to_unfreeze):
    layer_count = 0
    for name, param in model.named_parameters():
        if "conv1" in name or (name.startswith("conv_layers") and layer_count < num_conv_layers_to_unfreeze):
            param.requires_grad = True
        if name.startswith("conv_layers"):
            layer_count += 1


def load_pretrained_model(checkpoint_path, feature_size, model_params):
    print(f"Loading model from {checkpoint_path}")
    model = torch.load(checkpoint_path, map_location=device)
    model.to(device)
    print(f"Model loaded correctly")
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


run_id = get_next_run_id('model_results/training_results.csv')

def save_results(file_path, params, best_train_loss, best_test_loss):
    
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            #params ma senza "freeze_layers"
            params = {k: v for k, v in params.items() if k != "freeze_layers"}
            # Scrivi l'header solo se il file non esiste
            header = ["run_id"]+ list(params.keys()) + ['best_train_loss', 'best_test_loss',"freeze_layers"]
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
    for batch_index, batch in enumerate(tqdm(train_loader)):  #here train_loader is a DataLoader version specific for the graph data 
        #batch is the data for the current batch in the DataLoader
        # Use GPU
        batch.to(device)  
        # Reset gradients
        optimizer.zero_grad() 
        # Passing the node features and the connection info
        pred = model(batch.x.float(),                       #recall the forward function from the GNN model
                                batch.edge_attr.float(),
                                batch.edge_index, 
                                batch.batch) 
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
    for batch in test_loader:
        batch.to(device)  
        pred = model(batch.x.float(), 
                        batch.edge_attr.float(),
                        batch.edge_index, 
                        batch.batch) 
        loss = loss_fn(torch.squeeze(pred), batch.y.float())

         # Update tracking
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
    calculate_metrics(all_preds, all_labels, epoch, "test")
    log_conf_matrix(all_preds, all_labels, epoch)
    return running_loss/step

def log_conf_matrix(y_pred, y_true, epoch):
    # Log confusion matrix as image
    cm = confusion_matrix(y_true, y_pred)
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    classes = [str(c) for c in unique_classes]

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


#%% Run the training
from mango import Tuner
from config import HYPERPARAMETERS, BEST_PARAMETERS, SIGNATURE

def run_one_training(params, checkpoint_path=None):
    params = params[0]
    train_losses = []
    test_losses = []
    
    with mlflow.start_run() as run:
        for key in params.keys():
            mlflow.log_param(key, params[key])

        print("Loading dataset...")
        train_dataset = MoleculeDataset(root="data/", filename=params["train_file"]) 
        test_dataset = MoleculeDataset(root="data/", filename=params["test_file"], test=True)
        params["model_edge_dim"] = train_dataset[0].edge_attr.shape[1]
        print(f"params: {params}")

        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True) 
        test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=True)
        print(f"Train dataset size: {len(train_dataset)}")

        print("Loading model...")
        model_params = {k: v for k, v in params.items() if k.startswith("model_")}
        feature_size = train_dataset[0].x.shape[1]

        if checkpoint_path:
            model = load_pretrained_model(checkpoint_path, feature_size, model_params)
        else:
            model = GNN(feature_size=feature_size, model_params=model_params)
            model.to(device)

        print(f"Model parameters: {model_params}")
        print(f"Number of parameters: {count_parameters(model)}")
        mlflow.log_param("num_params", count_parameters(model))

        # Congela i primi livelli del modello
        num_layers_to_freeze = params["freeze_layers"]  # Numero di livelli da congelare (esempio)
        freeze_layers(model, num_layers_to_freeze)

        weight = torch.tensor([params["pos_weight"]], dtype=torch.float32).to(device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=params["learning_rate"],
                                    momentum=params["sgd_momentum"],
                                    weight_decay=params["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params["scheduler_gamma"])
        
        best_loss = 1000
        early_stopping_counter = 0
        for epoch in range(params["epochs"]): 
            if early_stopping_counter <= 10: 
                model.train()
                loss = train_one_epoch(epoch, model, train_loader, optimizer, loss_fn)
                train_losses.append(loss)
                print(f"Epoch {epoch} | Train Loss {loss}")
                mlflow.log_metric(key="Train loss", value=float(loss), step=epoch)

                model.eval()
                if epoch % 5 == 0:
                    loss = test(epoch, model, test_loader, loss_fn)
                    test_losses.append(loss)
                    print(f"Epoch {epoch} | Test Loss {loss}")
                    mlflow.log_metric(key="Test loss", value=float(loss), step=epoch)
                    
                    if float(loss) < best_loss:
                        best_loss = loss
                        mlflow.pytorch.log_model(model, "model", signature=SIGNATURE)
                        print(f"saving model with loss {best_loss}")
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1

                scheduler.step()
            else:
                print("Early stopping due to no improvement.")
                plot_loss_curves(train_losses, test_losses)
                save_results('model_results/training_results.csv', params, min(train_losses), min(test_losses))
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
    # mlflow.log_artifact(f"data/images/loss_curves_config_{run_id}.png")
    plt.show()
    

# %% Hyperparameter search
# print("Running hyperparameter search...")
config = dict()
# config["optimizer"] = "Bayesian"


BEST_PARAMETERS = {
   'batch_size': 64, 
   'learning_rate': 0.0014015474347793431, 
   'weight_decay': 0.002449693011254293, 
   'sgd_momentum': 0.8357800974603684, 
   'scheduler_gamma': 0.972996507445344,
   'pos_weight': 1.2, 
   'model_embedding_size': 256, 
   'model_attention_heads': 2, 
   'model_layers': 2, 
   'model_dropout_rate': 0.50040422634492724, 
   'model_top_k_ratio': 0.5995492263146199, 
   'model_top_k_every_n': 1, 
   'model_dense_neurons': 128,
    'epochs': 501,
    'train_file': "ETCM_train.csv",
    'test_file': "ETCM_test.csv",
    "freeze_layers": 44
   }

# # Create config using BEST_PARAMETERS
for key in BEST_PARAMETERS.keys():
    config[key] = BEST_PARAMETERS[key]


checkpoint_path = 'mlartifacts\\0\\92a28b05bde44860bddba6b12cc446d6_training_350epochs_balanced_train_set\\artifacts\model\data\model.pth'
running_loss = run_one_training([config], checkpoint_path=checkpoint_path)
# tuner = Tuner(BEST_PARAMETERS, 
#               objective=run_one_training,
#               conf_dict=config) 
# results = tuner.minimize()
