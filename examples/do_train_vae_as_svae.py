#!/usr/bin/env python3
"""Train TeacherVAE molecule generator."""
import os
import argparse
import train_vae


# From paccmann docs,
# https://github.com/PaccMann/paccmann_rl?tab=readme-ov-file#svae
#
# python ./code/paccmann_chemistry/examples/train_vae.py \
#    ./data/splitted_data/train_chembl_22_clean_1576904_sorted_std_final.smi \
#    ./data/splitted_data/test_chembl_22_clean_1576904_sorted_std_final.smi \
#    ./data/smiles_language_chembl_gdsc_ccle.pkl \
#    ./models/ \
#    ./code/paccmann_chemistry/examples/example_params.json \
#    svae

PACCMANN_DATA = os.getenv("PACCMANN_DATA", "./data")
PACCMANN_CODE = os.getenv("PACCMANN_CODE", "./code")
PACCMANN_MODELS = os.getenv("PACCMANN_MODELS", "./models")
TRAINING_NAME = "svae"

train_smiles_filepath = os.path.join(
    PACCMANN_DATA,
    # "splitted_data/train_chembl_22_clean_1576904_sorted_std_final.smi",  # original
    "splitted_data/train_chembl_22_clean_1576904_sorted_std_final_141921.smi"
)
test_smiles_filepath = os.path.join(
    PACCMANN_DATA,
    # "splitted_data/test_chembl_22_clean_1576904_sorted_std_final.smi",  # original
    "splitted_data/test_chembl_22_clean_1576904_sorted_std_final_15769.smi",
)
smiles_language_filepath = os.path.join(
    PACCMANN_DATA,
    "smiles_language_chembl_gdsc_ccle.pkl",
)
params_filepath = os.path.join(
    PACCMANN_CODE,
    "paccmann_chemistry/examples/example_params.json",
)
args = argparse.Namespace(
    train_smiles_filepath=train_smiles_filepath,
    test_smiles_filepath=test_smiles_filepath,
    smiles_language_filepath=smiles_language_filepath,
    model_path=PACCMANN_MODELS,
    params_filepath=params_filepath,
    training_name=TRAINING_NAME,
)

train_vae.main(parser_namespace=args)
