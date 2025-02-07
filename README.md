# Generate Personalized Connectomes
## 📌 Overview
This repository contains the code for conducting analyses described in the manuscript entitled "Deep generation of personalized connectomes based on individual attributes".
A conditional variational autoencoder is trained to generate individual connectomes from 194 personal measures, using the UK Biobank data.
## **📌 File description**
- **`matching_index/`** – Code for generating and evaluating connectomes with matching index benchmark
- **`setting/`** – Contain yaml settings for generating connectomes with variational autoencoder
- **`src/`** – Utility functions, such as model, loader, train epoch definitions, and more.
- **`s1_train_and_generate.py`** – Code for model training and personalized connectoem generation
- **`s2_sample_and_eval.py`** – Code for bootstrapping sample and evaluate generated interindividual variability
- **`s3_analysis.py/`** – Code for analyze and visualize results
---

## 📌 Data and checkpoints
Data and checkpoints are not available due to UK Biobank data poliocy. 
Users can access data via https://ams.ukbiobank.ac.uk/ams/ and train model with provided code.
