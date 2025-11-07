# 🧠 OptIForest: Replication and Dataset Extension Project

## 📘 Overview
This repository contains my replication and dataset-extension work for the **IJCAI 2023 paper**  
**“OptIForest: Optimal Isolation Forest for Anomaly Detection”** by *Xiang et al.*  
The project aims to:
- Reproduce the original OptIForest results on benchmark datasets.
- Construct new and enhanced datasets to test the robustness and generalisation of the model.

This work was completed as part of the **COMP8240 Research Project** at **Macquarie University (2025)**.



To set up the environment, first clone the OptIForest repository from GitHub using `git clone https://github.com/qwlv/OPTIFOREST-` and navigate into the project directory with `cd OPTIFOREST-`. Create a virtual environment using `python -m venv venv`, then activate it with `venv\Scripts\activate` on Windows or `source venv/bin/activate` on macOS/Linux. Once activated, install all required dependencies using `pip install -r requirements.txt`. If running in Google Colab, the same process applies — clone the repository using `!git clone https://github.com/qwlv/OPTIFOREST-`, change into the directory with `%cd OPTIFOREST-`, and install dependencies using `!pip install -r requirements.txt`. Ensure that Python version 3.8 or higher is used, along with the specified library versions: NumPy 1.24.4, Pandas 2.0.3, scikit-learn 1.3.2, Matplotlib 3.7.5, and tqdm 4.67.1.

To reproduce the original experiments from the IJCAI 2023 paper, use the `demo.py` script, which serves as the main driver of the OptIForest framework. This file replicates the reported results on the Arrhythmia (AD) and Ionosphere datasets. Run the commands `python demo.py --dataset ad --threshold 403 --branch 0` and `python demo.py --dataset ionosphere --threshold 403 --branch 0` to execute the tests. Here, the `--dataset` flag specifies which dataset to run, `--threshold 403` refers to the internal hyperparameter used in the paper, and `--branch 0` activates the L2-OPT configuration corresponding to the optimal branching factor (v = e). The program outputs key evaluation metrics, including AUC-ROC, AUC-PR, average training time, and testing time, computed over 15 independent runs.

In addition to the original datasets, several scripts are included to generate and prepare new datasets for extended experiments. The file `convertorcodeforthyroiddata.py` converts the raw Annthyroid dataset (`annthyroid_21feat_normalised.csv`) into a cleaned and standardized version (`annthyroid_clean.csv`) using KMeans clustering to assign binary anomaly labels. The `beer_dataset_generator.py` script creates the `beers_complex.csv` dataset, a realistic synthetic simulation of brewery production processes with correlated numerical features and injected anomalies. The `ionospherecsvconvertor.py` script extracts the Ionosphere dataset from its `.npz` format (as provided in ADBench) and converts it into a plain `.csv` file (`ionosphere.csv`) compatible with OptIForest. Lastly, the `newcomplexdatagenerator.py` script constructs the `synthetic_complex.csv` dataset, which is a high-dimensional synthetic dataset designed to test robustness under injected Gaussian noise, feature dropout, and distribution drift. Each script can be run independently to regenerate its respective dataset, ensuring full reproducibility and modular testing.


