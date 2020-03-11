import argparse
import os
import sys
sys.path.append('../')

import numpy as np
from moleculekit.molecule import Molecule

from util import get_file_list, get_atom_lines_and_labels_from_pdb, average_values, standardize_values


def get_parser():
    """
    Mapping of the prediction results onto PDB coordinate file.
    In the mode without processing, 10*log10(RMSF) values are mapped.

    Returns:
        Argument parser
    """
    parser = argparse.ArgumentParser(
        description='This script maps the 3DCNN prediction result onto PDB coordinate file as B factor.',
        usage=f"{os.path.basename(__file__)} -l <pdb_mrc.list> -p <pred.jbl> -o <output prefix name> (-n | -a) (-m)"
    )
    parser.add_argument(
        "-l", "--pdb_mrc_list", required=True, type=str,
        help="pdb_mrc.list file."
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
        '-o', '--output_prefix', type=str, default="md",
        help="output prefix name"
    )
    return parser.parse_args()


def get_md_vals(pdb_name, rmsf_xvg_name, gromacs_pdb):
    xyz, rmsf_list = get_atom_lines_and_labels_from_pdb(str(pdb_name), str(rmsf_xvg_name), str(gromacs_pdb))
    rmsf_list_arr = np.asarray(rmsf_list)
    log10rmsf_list_arr = np.log10(rmsf_list_arr)
    return log10rmsf_list_arr


def map_md_vals(mol, log10rmsf_list_arr, args):
    if args.normalize:
        norm_log10rmsf = standardize_values(log10rmsf_list_arr)
        mol.set('beta', np.round(norm_log10rmsf.flatten(), 5))
        mol.write(f'{args.output_prefix}_norm_model.pdb')
    elif args.average:
        norm_log10rmsf = standardize_values(log10rmsf_list_arr)
        avr_norm_log10rmsf = average_values(mol, norm_log10rmsf)
        mol.set('beta', np.round(avr_norm_log10rmsf.flatten(), 5))
        mol.write(f'{args.output_prefix}_avr_norm_model.pdb')
    else:
        mol.set('beta', np.round(10*np.log10(log10rmsf_list_arr).flatten(), 5))
        mol.write(f'{args.output_prefix}_model.pdb')


def main():
    args = get_parser()
    data_list = args.pdb_mrc_list
    pdb_name, map_name, rmsf_xvg_name, gromacs_pdb = get_file_list(data_list)
    md_log10rmsf = get_md_vals(str(pdb_name[0]), str(rmsf_xvg_name[0]), str(gromacs_pdb[0]))
    mol = Molecule(str(pdb_name[0]))
    mol.filter('protein')  # comment out for bb filtered bb
    map_md_vals(mol, md_log10rmsf, args)


if __name__ == "__main__":
    main()
