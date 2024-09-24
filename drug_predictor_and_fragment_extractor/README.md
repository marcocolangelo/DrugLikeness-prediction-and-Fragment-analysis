
# Drugness predictor and Fragments extractor

## Overview

This project implements a Graph Neural Network (GNN) model for classifying molecules in the context of Traditional Chinese Medicine drug-likeness. The model uses molecular graph representations and deep learning techniques to predict whether a given molecule is druglike. The model leverages graph-based learning for improved feature extraction and classification performance, focusing on molecular structure and interactions.

## Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Validation](#validation)
- [Prediction](#prediction)
- [Configuration](#configuration)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

The project is organized as follows:

- **`.gitignore`**: Specifies which files and directories to ignore in version control.
- **`anaconda_env.yaml`**: Environment configuration file for setting up the Conda environment.
- **`best_hyperparam.txt`**: File containing the best hyperparameters found during optimization.
- **`config.py`**: Configuration file to manage model parameters, paths, and training settings.
- **`continue_training.py`**: Script to resume training from a previously saved checkpoint.
- **`dashboard.py`**: Code to create a dashboard (possibly for monitoring experiments).
- **`data/`**: Directory containing the datasets.
- **`dataset.py`**: Handles the dataset loading and pre-processing functions.
- **`dataset_featurizer.py`**: Functions for featurizing the molecular graphs.
- **`dataset_featurizer_no_parall.py`**: A version of the featurizer without parallelization support.
- **`model.py`**: Defines the core GNN model used for classification.
- **`MSSGAT/`**: A sub-module or additional architecture for handling graph attention networks.
- **`my_code/`**: Custom scripts and experiments for advanced analysis (see below for details).
- **`mlartifacts/`, `mlflow.db`, `mlruns/`**: Directories and files related to MLFlow for experiment tracking.
- **`model_results/`**: Directory for saving results from trained models.
- **`requirements.txt`**: List of required dependencies to run the project.
- **`train.py`**: The main training script for the GNN model.
- **`tree_transf.py`**: Implements tree-based transformations, likely for hierarchical GNN representations.
- **`utils.py`**: Utility functions for logging, data handling, and miscellaneous operations.
- **`validation_train.py`**: Script to validate the model and tune hyperparameters.
- **`OLD_README.md`**: A prior version of the README file.
- **`output_validation.txt`**: Validation output file.
- **`predict.py`**: Script for generating predictions on new data.
- **`prove_varie.ipynb`**: Jupyter Notebook for exploratory data analysis and testing.
- **`trash_to_delete/`**: Temporary files or scripts not actively used.

### Files in `my_code/`

- **`conc_fp_HDBSCAN_uniqueFrag_high_att_frags_analysis.py`**: Script that performs clustering using the HDBSCAN algorithm on high-attention fragments and fingerprint concatenation.
- **`fine_tuning_freezing_layers.py`**: This script focuses on fine-tuning the model by freezing specific layers during the training process.
- **`HDBSCAN_GPU_uniqueFrag_high_att_frags_analysis.py`**: A GPU-accelerated version of the HDBSCAN algorithm for analyzing high-attention fragments with unique fingerprints.
- **`HDBSCAN_uniqueFrag_high_att_frags_analysis.py`**: Runs the HDBSCAN clustering algorithm to analyze unique high-attention fragments without GPU acceleration.
- **`highlight_mol.py`**: A utility script used to highlight specific molecules or fragments in the visualization process.
- **`high_att_frags_analysis.py`**: Performs analysis on fragments that receive high attention scores during the GNN model's prediction process.
- **`oversample_data.py`**: Implements oversampling techniques to balance the dataset, particularly useful for addressing class imbalance issues.
- **`study_model_layers.py`**: Analyzes the different layers of the GNN model, examining how information is processed across the layers.
- **`test_building_tree.py`**: A test script for building molecular trees, likely used in conjunction with hierarchical molecular graph processing.
- **`TransfGNN.py`**: Implements the Transformer-based GNN architecture, which applies attention mechanisms to graph-structured data.
- **`tree_building.py`**: Contains functions or classes for constructing trees from molecular graphs, a step necessary for certain GNN architectures.
- **`tree_path.py`**: Handles path-related computations in molecular trees, possibly tracking the connection paths within the graph.
- **`unify_high_mol.py`**: Unifies high-attention molecules into a common format or structure for analysis.
- **`unify_high_mol_2.py`**: A second version of the `unify_high_mol.py` script, likely with improvements or alternative approaches.

## Prerequisites

1. **Anaconda**: Make sure you have [Anaconda](https://www.anaconda.com/products/distribution) installed.
2. **Python 3.7+**: The code is tested on Python versions 3.7 and above.
3. **CUDA** (optional but recommended): If using GPU for faster training, ensure CUDA is installed.

## Installation

### Step 1: Create Conda Environment

You can create a Conda environment with all the necessary dependencies using the provided `anaconda_env.yaml` file:

```bash
conda env create -f anaconda_env.yaml
conda activate gnn-hiv-example
```

### Step 2: Manual Dependency Installation

Alternatively, you can install the required packages manually by running:

```bash
pip install -r requirements.txt
```

### Step 3: Set Up MLFlow

MLFlow is used for experiment tracking. Ensure MLFlow is set up and configure the logging directory by running:

```bash
mlflow ui
```

## Usage

### 1. **Training the Model**

To train the model, run the `train.py` script. This will initialize the GNN model, load the dataset, and begin training.

```bash
python train.py
```

Results such as training logs and checkpoints will be stored in the `mlruns/` directory, and the best model parameters will be saved.

### 2. **Validation**

To validate the model on a test dataset or validate using cross-validation, you can run the `validation_train.py` script:

```bash
python validation_train.py
```

This script will evaluate the model's performance and tune hyperparameters using the validation set.

### 3. **Prediction**

To generate predictions using the trained model, execute the `predict.py` script with the required input data:

```bash
python predict.py --input <input_csv_file> --output <output_csv_file>
```

Make sure the input file is correctly formatted as required by the model.

## Configuration

### Model and Hyperparameters

Model configurations and hyperparameters can be adjusted in the `config.py` file. Here, you can modify:

- Learning rate
- Batch size
- Optimizer settings
- Model architecture parameters (number of layers, heads, etc.)
- Training and validation split ratios

Additionally, for hyperparameter optimization, the script `validation_train.py` will use Optuna to find the best hyperparameters automatically.

## Results

The best trained model is mlartifacts\0\388693edcfd747f6ae7f2edb65d65cd2 with metrics:

F1 Score: 0.8412029744679292
Accuracy: 0.8323276862381063
Precision: 0.7995264953694032
Recall: 0.8874632864430361
ROC AUC: 0.8322807296259775


To review experiment details and visualize the results, launch the MLFlow UI:

```bash
mlflow ui
```

This will allow you to view metrics, graphs, and comparisons between different experiment runs.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new feature branch (`git checkout -b feature/my-new-feature`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/my-new-feature`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
