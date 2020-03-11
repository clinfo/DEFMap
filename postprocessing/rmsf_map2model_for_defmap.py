import argparse
import os
import sys
import joblib
sys.path.append('../')

import numpy as np
from moleculekit.molecule import Molecule
from pathlib import Path

from util import get_em_map, get_voxel, standardize_values, average_values


def get_parser():
    """
    Mapping of the prediction results onto PDB coordinate file.
    In the mode without processing, 10*log10(RMSF) values are mapped.

    Returns:
        Argument parser
    """
    parser = argparse.ArgumentParser(
        description='This script maps the 3DCNN prediction result onto PDB coordinate file as B factor.',
        usage=f"{os.path.basename(__file__)} -l <pdb_mrc.list> -p <pred.jbl> -o <output prefix name> (-n | -a)"
    )
    parser.add_argument(
        "-l", "--pdb_mrc_list", required=True, type=str,
        help="pdb_mrc.list file."
    )
    parser.add_argument(
        "-p", "--pred",  type=str,
        help=".jbl for RMSF prediction calculated by 3dcnn_main.py."
    )
    parser.add_argument(
        '-n', '--normalize', action="store_true",
        help="map normalized predicted values and plot correlation."
    )
    parser.add_argument(
        '-a', '--average', action="store_true",
        help="map residue-average 'normalized' predicted values and plot correlation."
    )
    parser.add_argument(
        '-o', '--output_prefix', type=str, default="defmap",
        help="output prefix name"
    )
    return parser.parse_args()


def get_file_list(data_list):
    path = Path(data_list)
    with open(path, 'r') as f:
        lines = f.readlines()
    pdb_names, map_names = [], []
    for l in lines:
        p, m = l.split()
        pdb_names.append(path.with_name(p))
        map_names.append(path.with_name(m))
    return pdb_names, map_names


def get_rmsf_from_pred(mol, pred, info):
    xyz = mol.get('coords')
    xyz_vox = get_voxel(xyz, info)
    xyz_vox_keys = []
    for c in xyz_vox:
        xyz_vox_keys.append(','.join(list(map(str, map(int, reversed(c.tolist()))))))
    pred_dic = joblib.load(pred)
    log10rmsf_val = [pred_dic[k][0] for k in xyz_vox_keys if k in pred_dic.keys()]
    log10rmsf_val_arr = np.asarray(log10rmsf_val)
    return log10rmsf_val_arr


def map_pred_vals(mol, pred, info, args):
    pred_log10rmsf = get_rmsf_from_pred(mol, pred, info)
    if args.normalize:
        norm_log10rmsf = standardize_values(pred_log10rmsf)
        mol.set('beta', np.round(norm_log10rmsf.flatten(), 5))
        mol.write(f'{args.output_prefix}_norm_model.pdb')
    elif args.average:
        norm_log10rmsf = standardize_values(pred_log10rmsf)
        avr_norm_log10rmsf = average_values(mol, norm_log10rmsf)
        mol.set('beta', np.round(avr_norm_log10rmsf.flatten(), 5))
        mol.write(f'{args.output_prefix}_avr_norm_model.pdb')
    else:
        mol.set('beta', np.round(10 * np.log10(pred_log10rmsf).flatten(), 5))
        mol.write(f'{args.output_prefix}_model.pdb')


def main():
    args = get_parser()
    pred = args.pred
    data_list = args.pdb_mrc_list
    pdb_name, map_name = get_file_list(data_list)
    em_arr, info = get_em_map(map_name[0])
    mol = Molecule(str(pdb_name[0]))
    mol.filter('protein')  # comment out for bb filtered bb
    map_pred_vals(mol, pred, info, args)
 

if __name__ == "__main__":
    main()
