#!/usr/bin/env python3
"""Train TeacherVAE molecule generator."""
import argparse
import os

import selfies
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

# ----------------------------------------------------------------------------
# Prevents error from "hypervalent" molecules, such as nitrogen with 5 bonds,
# which are present in the data:
# ./data/splitted_data/test_chembl_22_clean_1576904_sorted_std_final
# For example, selfies default on SMILES: 
#        "CCOC(=O)CC(OC1OC2OC3(C)CCC4C(C)CCC(C1C)C24OO3)c1ccc(cc1)N(=O)=O"
# causes "EncoderError, selfies.exceptions.EncoderError: input violates the currently-set semantic constraints"
#        "Errors:""
#        "[N with 5 bond(s) - a max. of 3 bond(s) was specified]""
# ----------------------------------------------------------------------------
# selfies.set_semantic_constraints(selfies.get_preset_constraints("hypervalent"))

train_smiles_filepath = os.path.join(
    PACCMANN_DATA,
    # "splitted_data/train_chembl_22_clean_1576904_sorted_std_final.smi",  # original
    # "splitted_data/train_chembl_22_clean_1576904_sorted_std_final_141921.smi",  # 10% subset
    # "splitted_data/train_chembl_22_clean_1576904_sorted_std_final-filt.smi",  # filter_invalid_smiles
    # "splitted_data/train_chembl_22_clean_1576904_sorted_std_final-filt2.smi",  # sanitize=True in filter_invalid_smi rdkit call
    "splitted_data/train_chembl_22_clean_1576904_sorted_std_final-filt-self.smi",
)
test_smiles_filepath = os.path.join(
    PACCMANN_DATA,
    # "splitted_data/test_chembl_22_clean_1576904_sorted_std_final.smi",  # original
    # "splitted_data/test_chembl_22_clean_1576904_sorted_std_final_15769.smi",
    # "splitted_data/test_chembl_22_clean_1576904_sorted_std_final-filt.smi",  # filter_invalid_smiles
    # "splitted_data/test_chembl_22_clean_1576904_sorted_std_final-filt2.smi",  # sanitize=True in filter_invalid_smi rdkit call
    "splitted_data/test_chembl_22_clean_1576904_sorted_std_final-filt-self.smi",
)
smiles_language_filepath = os.path.join(
    PACCMANN_DATA,
    "smiles_language_chembl_gdsc_ccle.pkl",  # Legacy pkl form
    # "smiles_language_chembl_gdsc_ccle/",  # New directory form
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
