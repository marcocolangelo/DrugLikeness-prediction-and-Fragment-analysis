import torch
from model import GNN  # Assicurati di importare il tuo modello correttamente

def check_model_layers(model):
    # Verifica se il modello ha il metodo named_parameters
    if hasattr(model, 'named_parameters'):
        print("Il comando 'model.named_parameters()' esiste nel modello.")
        print("Elenca i parametri del modello:")
        for name, param in model.named_parameters():
            print(f"Nome: {name}, Dimensione: {param.size()}")
        # Conta il numero di strati nel modello
        num_layers = sum(1 for _ in model.named_parameters())
        print(f"Il modello ha {num_layers} strati (layers).")
    else:
        print("Il comando 'model.named_parameters()' NON esiste nel modello.")

# Imposta i parametri del modello (esempio)
feature_size = 256
model_params = {
    'model_embedding_size': 256,
    'model_attention_heads': 2,
    'model_layers': 2,
    'model_dropout_rate': 0.2,
    'model_top_k_ratio': 0.6,
    'model_top_k_every_n': 1,
    'model_dense_neurons': 128,
    'model_edge_dim': 11
}

# Crea un'istanza del modello
model = GNN(feature_size=feature_size, model_params=model_params)
model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

# Esegui la verifica
check_model_layers(model)
