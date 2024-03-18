#!/usr/bin/env python3
"""Train TeacherVAE molecule generator."""
import argparse
from collections import Counter
import json
import logging
import os
import sys
from time import time
from types import SimpleNamespace
from paccmann_chemistry.utils import (
    collate_fn, get_device, disable_rdkit_logging
)
from paccmann_chemistry.models.vae import (
    StackGRUDecoder, StackGRUEncoder, TeacherVAE
)
from paccmann_chemistry.models.training import train_vae
from paccmann_chemistry.utils.hyperparams import SEARCH_FACTORY
from pytoda.datasets import SMILESDataset
from pytoda.datasets import SMILESTokenizerDataset
from pytoda.smiles.smiles_language import SMILESLanguage
from torch.utils.tensorboard.writer import SummaryWriter
import torch
from pytoda.smiles import transforms


# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('training_vae')

# yapf: disable
parser = argparse.ArgumentParser(description='Chemistry VAE training script.')
parser.add_argument(
    'train_smiles_filepath', type=str,
    help='Path to the train data file (.smi).'
)
parser.add_argument(
    'test_smiles_filepath', type=str,
    help='Path to the test data file (.smi).'
)
parser.add_argument(
    'smiles_language_filepath', type=str,
    help='Path to SMILES language object.'
)
parser.add_argument(
    'model_path', type=str,
    help='Directory where the model will be stored.'
)
parser.add_argument(
    'params_filepath', type=str,
    help='Path to the parameter file.'
)
parser.add_argument(
    'training_name', type=str,
    help='Name for the training.'
)
# yapf: enable

def update_smiles_language(smiles_language, params):
    """
    Adapts SMILESLanguage objects from earlier versions (0.0.1 or 0.1.1) of
    pytoda, typically loaded from pickle, to include attributes required by
    later version pytoda (1.1.3).

    Also applies the same defaults to these as in the original code for
    `smiles_train_data` and `smiles_test_data`
    """
    # From pytoda in paccmann_datasets 1.1.3
    attrs = [
        "canonical", "augment", "kekulize", "all_bonds_explicit", "selfies",
        "sanitize", "all_hs_explicit", "remove_bonddir", "remove_chirality",
        "randomize", "add_start_and_stop", "padding", "padding_length"
    ]

    # Precompute values to fill in attributes missing from the pytoda 0.0.1
    # SMILESLanguage object load'ed from pickle, and that are required in
    # pytoda 1.1.3
    values = dict(
        padding=False,
        selfies=params.get('selfies', False),
        add_start_and_stop=params.get('add_start_stop_token', True),
        augment=params.get('augment_smiles', False),
        canonical=params.get('canonical', False),
        kekulize=params.get('kekulize', False),
        all_bonds_explicit=params.get('all_bonds_explicit', False),
        all_hs_explicit=params.get('all_hs_explicit', False),
        remove_bonddir=params.get('remove_bonddir', False),
        remove_chirality=params.get('remove_chirality', False),
        sanitize=params.get('sanitize', True),
        randomize=params.get('randomize', False),
        padding_length=params.get('padding_length', None),
    )

    # If smiles_language lacks a required attribute,
    # set it to be the params value, if any,
    # otherwise set to the best default I could guess:
    # * False for most params,
    # * True for `add_start_and_stop`, and `sanitize`.
    # * None for `padding_length`
    for attr in attrs:
        if not hasattr(smiles_language, attr):
            value = values.get(attr, None)
            setattr(smiles_language, attr, value)
        if attr not in params:
            params[attr] = getattr(smiles_language, attr)

    # Add to earlier-vintage SMILESLanguage 3 attributes required by logic
    # associated with pytoda 1.1.3 and later-version SMILESLanguage objects:
    # .transform_smiles, .transform_encoding, and .token_count
    smiles_language.transform_smiles = transforms.Compose([])
    smiles_language.transform_encoding = transforms.Compose([])
    smiles_language.token_count = Counter()


def _set_token_len_fn(self, add_start_and_stop):
    """
    Defines a Callable that given a sequence of naive tokens, i.e. before
    applying the encoding transforms, computes the number of
    implicit tokens after transforms (implicit because it's the
    number of token indexes, not actual tokens).
    """
    if add_start_and_stop:
        self._get_total_number_of_tokens_fn = (
            SMILESLanguage.__get_total_number_of_tokens_with_start_stop_fn
        )
    else:
        self._get_total_number_of_tokens_fn = len


def reset_initial_transforms(self):
    """
    Originally: Reset smiles and token indexes transforms as on initialization.

    This function, borrowed from a method in pytoda 1.1.3, seems to apply
    only to instances of SMILESTokenizer class, not to regular SMILESLanguage
    instances.
    """
    self.transform_smiles = transforms.compose_smiles_transforms(
        self.canonical,
        self.augment,
        self.kekulize,
        self.all_bonds_explicit,
        self.all_hs_explicit,
        self.remove_bonddir,
        self.remove_chirality,
        self.selfies,
        self.sanitize,
    )
    self.transform_encoding = transforms.compose_encoding_transforms(
        self.randomize,
        self.add_start_and_stop,
        self.start_index,
        self.stop_index,
        self.padding,
        self.padding_length,
        self.padding_index,
    )
    self._set_token_len_fn(self.add_start_and_stop)


def main(parser_namespace):
    try:
        device = get_device()
        disable_rdkit_logging()
        # read the params json
        params = dict()
        with open(parser_namespace.params_filepath) as f:
            params.update(json.load(f))

        # get params
        train_smiles_filepath = parser_namespace.train_smiles_filepath
        test_smiles_filepath = parser_namespace.test_smiles_filepath
        smiles_language_filepath = (
            parser_namespace.smiles_language_filepath
            if parser_namespace.smiles_language_filepath.lower() != 'none' else
            None
        )

        model_path = parser_namespace.model_path
        training_name = parser_namespace.training_name

        writer = SummaryWriter(f'logs/{training_name}')

        logger.info(f'Model with name {training_name} starts.')

        model_dir = os.path.join(model_path, training_name)
        log_path = os.path.join(model_dir, 'logs')
        val_dir = os.path.join(log_path, 'val_logs')
        os.makedirs(os.path.join(model_dir, 'weights'), exist_ok=True)
        os.makedirs(os.path.join(model_dir, 'results'), exist_ok=True)
        os.makedirs(log_path, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        # Load SMILES language
        smiles_language = None
        if smiles_language_filepath is not None:
            smiles_language = SMILESLanguage.load(smiles_language_filepath)
            update_smiles_language(smiles_language, params)

        logger.info('Smiles filepath: %s', train_smiles_filepath)
        
        # smiles_language.reset_initial_transforms = SMILESLanguage.reset_initial_transforms.__get__(smiles_language)
        # smiles_language._set_token_len_fn = _set_token_len_fn.__get__(smiles_language)
        # smiles_language.reset_initial_transforms = reset_initial_transforms.__get__(smiles_language)
        # smiles_language.reset_initial_transforms()

        # create SMILES eager dataset
        smiles_train_data = SMILESTokenizerDataset(
            train_smiles_filepath,
            smiles_language=smiles_language,
            padding=False,
            selfies=params.get('selfies', False),
            add_start_and_stop=params.get('add_start_stop_token', True),
            augment=params.get('augment_smiles', False),
            canonical=params.get('canonical', False),
            kekulize=params.get('kekulize', False),
            all_bonds_explicit=params.get('all_bonds_explicit', False),
            all_hs_explicit=params.get('all_hs_explicit', False),
            remove_bonddir=params.get('remove_bonddir', False),
            remove_chirality=params.get('remove_chirality', False),
            backend='lazy',
            device=None,
        )
        smiles_test_data = SMILESTokenizerDataset(
            test_smiles_filepath,
            smiles_language=smiles_language,
            padding=False,
            selfies=params.get('selfies', False),
            add_start_and_stop=params.get('add_start_stop_token', True),
            augment=params.get('augment_smiles', False),
            canonical=params.get('canonical', False),
            kekulize=params.get('kekulize', False),
            all_bonds_explicit=params.get('all_bonds_explicit', False),
            all_hs_explicit=params.get('all_hs_explicit', False),
            remove_bonddir=params.get('remove_bonddir', False),
            remove_chirality=params.get('remove_chirality', False),
            backend='lazy',
            device=None,
        )

        if smiles_language_filepath is None:
            smiles_language = smiles_train_data.smiles_language
            smiles_language.save(
                os.path.join(model_path, f'{training_name}.lang')
            )
        else:
            smiles_language_filename = os.path.basename(smiles_language_filepath)
            smiles_language.save(
                os.path.join(model_dir, smiles_language_filename)
            )

        params.update(
            {
                'vocab_size': smiles_language.number_of_tokens,
                'pad_index': smiles_language.padding_index
            }
        )

        vocab_dict = smiles_language.index_to_token
        params.update(
            {
                'start_index':
                    list(vocab_dict.keys())
                    [list(vocab_dict.values()).index('<START>')],
                'end_index':
                    list(vocab_dict.keys())
                    [list(vocab_dict.values()).index('<STOP>')]
            }
        )

        if params.get('embedding', 'learned') == 'one_hot':
            params.update({'embedding_size': params['vocab_size']})

        with open(os.path.join(model_dir, 'model_params.json'), 'w') as fp:
            json.dump(params, fp)

        # create DataLoaders
        train_data_loader = torch.utils.data.DataLoader(
            smiles_train_data,
            batch_size=params.get('batch_size', 64),
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=True,
            pin_memory=params.get('pin_memory', True),
            num_workers=params.get('num_workers', 8)
        )
        # train_data_loader.dataset._dataset = SimpleNamespace()
        # train_data_loader.dataset._dataset.selfies = params.get('selfies', False)
        test_data_loader = torch.utils.data.DataLoader(
            smiles_test_data,
            batch_size=params.get('batch_size', 64),
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=True,
            pin_memory=params.get('pin_memory', True),
            num_workers=params.get('num_workers', 8)
        )
        # test_data_loader.dataset._dataset = SimpleNamespace()
        # test_data_loader.dataset._dataset.selfies = params.get('selfies', False)
        # initialize encoder and decoder
        gru_encoder = StackGRUEncoder(params).to(device)
        gru_decoder = StackGRUDecoder(params).to(device)
        gru_vae = TeacherVAE(gru_encoder, gru_decoder).to(device)
        # TODO I haven't managed to get this to work. I will leave it here
        # if somewant (or future me) wants to give it a look and get the
        # tensorboard graph to work
        # if writer and False:
        #     gru_vae.set_batch_mode('padded')
        #     dummy_input = torch.ones(smiles_train_data[0].shape)
        #     dummy_input = dummy_input.unsqueeze(0).to(device)
        #     writer.add_graph(gru_vae, (dummy_input, dummy_input, dummy_input))
        #     gru_vae.set_batch_mode(params.get('batch_mode'))
        logger.info('\n****MODEL SUMMARY***\n')
        for name, parameter in gru_vae.named_parameters():
            logger.info(f'Param {name}, shape:\t{parameter.shape}')
        total_params = sum(p.numel() for p in gru_vae.parameters())
        logger.info(f'Total # params: {total_params}')

        loss_tracker = {
            'test_loss_a': 10e4,
            'test_rec_a': 10e4,
            'test_kld_a': 10e4,
            'ep_loss': 0,
            'ep_rec': 0,
            'ep_kld': 0
        }

        # train for n_epoch epochs
        logger.info(
            'Model creation and data processing done, Training starts.'
        )
        decoder_search = SEARCH_FACTORY[
            params.get('decoder_search', 'sampling')
        ](
            temperature=params.get('temperature', 1.),
            beam_width=params.get('beam_width', 3),
            top_tokens=params.get('top_tokens', 5)
        )  # yapf: disable

        if writer:
            pparams = params.copy()
            pparams['training_file'] = train_smiles_filepath
            pparams['test_file'] = test_smiles_filepath
            pparams['language_file'] = smiles_language_filepath
            pparams['model_path'] = model_path
            pparams = {
                k: v if v is not None else 'N.A.'
                for k, v in params.items()
            }
            pparams['training_name'] = training_name
            from pprint import pprint
            pprint(pparams)
            writer.add_hparams(hparam_dict=pparams, metric_dict={})

        for epoch in range(params['epochs'] + 1):
            t = time()
            loss_tracker = train_vae(
                epoch,
                gru_vae,
                train_data_loader,
                test_data_loader,
                smiles_language,
                model_dir,
                search=decoder_search,
                optimizer=params.get('optimizer', 'adadelta'),
                lr=params['learning_rate'],
                kl_growth=params['kl_growth'],
                input_keep=params['input_keep'],
                test_input_keep=params['test_input_keep'],
                generate_len=params['generate_len'],
                log_interval=params['log_interval'],
                save_interval=params['save_interval'],
                eval_interval=params['eval_interval'],
                loss_tracker=loss_tracker,
                logger=logger,
                # writer=writer,
                batch_mode=params.get('batch_mode')
            )
            logger.info(f'Epoch {epoch}, took {time() - t:.1f}.')

        logger.info(
            'OVERALL: \t Best loss = {0:.4f} in Ep {1}, '
            'best Rec = {2:.4f} in Ep {3}, '
            'best KLD = {4:.4f} in Ep {5}'.format(
                loss_tracker['test_loss_a'], loss_tracker['ep_loss'],
                loss_tracker['test_rec_a'], loss_tracker['ep_rec'],
                loss_tracker['test_kld_a'], loss_tracker['ep_kld']
            )
        )
        logger.info('Training done, shutting down.')
    except Exception:
        logger.exception('Exception occurred while running train_vae.py.')


if __name__ == '__main__':
    args = parser.parse_args()
    main(parser_namespace=args)
