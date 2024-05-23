import torch
import random
import argparse
import numpy as np
import ipdb as pdb
import os, pwd, yaml
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from caring.modules.stationary import StationaryProcess
from caring.tools.utils import load_yaml, setup_seed
from caring.datasets.sim_dataset import StationaryDataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import warnings
warnings.filterwarnings('ignore')

def main(args):

    assert args.exp is not None, "FATAL: "+__file__+": You must specify an exp config file (e.g., *.yaml)"
    
    current_user = pwd.getpwuid(os.getuid()).pw_name
    script_dir = os.path.dirname(__file__)
    rel_path = os.path.join('../caring/configs', 
                            '%s.yaml'%args.exp)
    abs_file_path = os.path.join(script_dir, rel_path)
    cfg = load_yaml(abs_file_path)
    print("######### Configuration #########")
    print(yaml.dump(cfg, default_flow_style=False))
    print("#################################")

    pl.seed_everything(args.seed)
    # pdb.set_trace()

    data = StationaryDataset(directory=cfg['ROOT'],
                             transition=cfg['DATASET'])

    num_validation_samples = cfg['VAE']['N_VAL_SAMPLES']
    train_data, val_data = random_split(data, [len(data)-num_validation_samples, num_validation_samples])

    train_loader = DataLoader(train_data, 
                              batch_size=cfg['VAE']['TRAIN_BS'], 
                              pin_memory=cfg['VAE']['PIN'],
                              num_workers=cfg['VAE']['CPU'],
                              drop_last=False,
                              shuffle=True)

    val_loader = DataLoader(val_data, 
                            batch_size=cfg['VAE']['VAL_BS'], 
                            pin_memory=cfg['VAE']['PIN'],
                            num_workers=cfg['VAE']['CPU'],
                            shuffle=False)

    model = StationaryProcess(input_dim=cfg['VAE']['INPUT_DIM'],
                              length=cfg['VAE']['LENGTH'],
                              z_dim=cfg['VAE']['LATENT_DIM'], 
                              lag=cfg['VAE']['LAG'],
                              hidden_dim=cfg['VAE']['ENC']['HIDDEN_DIM'],
                              trans_prior=cfg['VAE']['TRANS_PRIOR'],
                              lr=cfg['VAE']['LR'],
                              beta=cfg['VAE']['BETA'],
                              gamma=cfg['VAE']['GAMMA'],
                              decoder_dist=cfg['VAE']['DEC']['DIST'],
                              decoder_side_truez=cfg['VAE']['DEC']['SIDE_TRUEZ'],
                              decoder_side_hatz=cfg['VAE']['DEC']['SIDE_HATZ'],
                              decoder_side_truex=cfg['VAE']['DEC']['SIDE_TRUEX'],
                              encoder_side_truez=cfg['VAE']['ENC']['SIDE_TRUEZ'],
                              encoder_side_truex=cfg['VAE']['ENC']['SIDE_TRUEX'],
                              correlation=cfg['MCC']['CORR'])
    
    log_dir = os.path.join(cfg["LOG"], current_user, args.exp)

    checkpoint_callback = ModelCheckpoint(monitor='val_mcc',
                                          save_top_k=1, 
                                          mode='max')

    early_stop_callback = EarlyStopping(monitor="val_mcc", 
                                        min_delta=0.00, 
                                        patience=50, 
                                        verbose=False, 
                                        mode="max")

    trainer = pl.Trainer(default_root_dir=log_dir,
                         gpus=cfg['VAE']['GPU'], 
                         val_check_interval = cfg['MCC']['FREQ'],
                         max_epochs=cfg['VAE']['EPOCHS'],
                         callbacks=[checkpoint_callback, early_stop_callback])

    # Train the model
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-e',
        '--exp',
        type=str
    )

    argparser.add_argument(
        '-s',
        '--seed',
        type=int,
        default=770
    )

    args = argparser.parse_args()
    main(args)
