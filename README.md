# Molecular Fragment Extraction Using Transformer-based Graph Neural Networks (GNN) on Traditional Chinese Medicine molecules

## Project Overview

This project focuses on the development of a deep learning architecture capable of processing multiple molecular structures as input and extracting chemically relevant fragments. The goal is to identify molecular substructures that are not only common across multiple input molecules but are also chemically significant.

The model employs **Graph Neural Networks (GNN)** to process molecular graphs, leveraging **Transformer-based convolutions** to compute attention scores on the graph. These scores allow for the identification of important molecular substructures, which can then be analyzed and clustered for further interpretation.

## Objectives

- Predicting whether a molecule is drug-like or not.
- Use attention mechanisms to highlight parts of the molecule most influential to the model's decision-making process.
- Extract common and chemically relevant molecular fragments from a set of input molecules.
- Employ clustering techniques to group similar molecular fragments based on structural similarity and positioning within the molecule.

## Key Features

- **Graph Neural Network (GNN) architecture**: The model processes molecular graphs using layers such as TransformerConv, TopKPooling, and BatchNorm1d.
- **Attention Mechanisms**: The model highlights relevant substructures by assigning attention scores to nodes and edges in the graph.
- **Molecular Fragmentation**: Decomposes molecules into chemically meaningful fragments, focusing on the preservation of key substructures like aromatic rings.
- **SMARTS Fingerprinting**: Converts molecular fragments into SMARTS representation to capture general structural patterns for comparison and clustering.
- **Clustering of Fragments**: Groups extracted fragments based on their structural similarity using methods like Tanimoto similarity.

## Installation

This project uses Python 3.8+ and depends on several key libraries, including PyTorch, RDKit, DeepChem, and DGL. To set up the environment:

1. Clone this repository:

   ```bash
   git clone https://github.com/marcocolangelo/DrugLikeness-prediction-and-Fragment-analysis
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Set up a conda environment:

   ```bash
   conda env create -f environment.yml
   conda activate molecule-extraction
   ```

## Usage

### Preprocessing

The preprocessing stage generates molecular trees and graph structures from SMILES notation. Use the `MoleculeDataset` class to process and featurize molecules before training.


### DrugLikeness-prediction-and-Fragment-analysis
This is the main folder, where the file for training and prediction are saved.

#### Training the Model  

To train the GNN on your dataset, use the `train.py` script:

```bash
python train.py 
```

You can adjust hyperparameters in the config file, such as learning rate, batch size, and the number of GNN layers.


#### Continue training

If you started training and want to load the relative checkpoint to continue it, you can execute the `continue_training.py` file.

```bash
python continue_training.py 
```


#### Prediction

You can use a pre-trained model when executing the file predict.py

```bash
python predict.py 
```

You can modify the flags `deep-search` and `final`.
- `deep-search` : activates the modality that exploits a BTS for capturing bigger relevant substructures instead of single relevant fragments.
- `final` : performs the predictions on the TMC dataset we uploaded in the repository.
- 


### Post-processing

It is where the final analysis of the extracted fragments is performed. 

It performs a deduplication operation in order to find the unique occurrences and the total occurrences for each fragment, computes the fingerprints and applies the HDBSCAN algorithm to them. Everything is saved in dedicated files.

Once you have obtained the predictions and high-attention fragments files from the prediction script, you can execute `unique_frag_high_att_frags_analysis.py` to gather all this information on the collected fragments. 

It is compatible even with fragments obtained with the deep_search flag enabled.

## Contributing

Contributions to the project are welcome! Please fork the repository and submit a pull request with your improvements. Make sure to update tests as appropriate.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **MSSGAT**: Special thanks to the authors of *Multi-Scale Self-Supervised Graph Attention Networks (MSSGAT)* for their contributions to the attention mechanisms applied in this project.
- **DeepFindr**: This code has taken inspiration from the project you can find through the following link https://github.com/deepfindr/gnn-project by the user DeepFindr
- **PyTorch Geometric** for providing the necessary tools for GNNs.
- **RDKit** and **DeepChem** for molecular processing and feature extraction.
