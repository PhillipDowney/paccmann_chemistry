import argparse
import logging
import os
import sys
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import selfies

# from pytoda.preprocessing.smi import filter_invalid_smi
from pytoda.files import read_smi
from pytoda.smiles.transforms import Canonicalization


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('filter_invalid_smi')

parser = argparse.ArgumentParser(description='Chemistry SELFIES encoder-error filter')
parser.add_argument(
    'input_filepath', type=str,
    help='Path to the input data file (.smi)'
)
parser.add_argument(
    'output_filepath', type=str,
    help='Path to the output data file (.smi)'
)


def main(args):
    filter_invalid_smi(args.input_filepath, args.output_filepath)


def filter_invalid_smi(
    input_filepath: str, output_filepath: str, chunk_size: int = 100000,
    sanitize: bool = False
):
    """
    Execute chunked invalid SMILES filtering in a .smi file.

    Args:
        input_filepath (str): path to the .smi file to process.
        output_filepath (str): path where to store the filtered .smi file.
        chunk_size (int): size of the SMILES chunk. Defaults to 100000.
        sanitize (bool): if True
    """
    def encodes_ok(smiles):
        try:
            _ = selfies.encoder(smiles)
        except selfies.EncoderError:
            return False
        else:
            return True

    for chunk in read_smi(input_filepath, chunk_size=chunk_size):
        pd.DataFrame(
            [
                [row['SMILES'], index]
                for index, row in chunk.iterrows()
                if encodes_ok(row['SMILES'])
            ]
        ).to_csv(output_filepath, index=False, header=False, mode='a', sep='\t')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args=args)
