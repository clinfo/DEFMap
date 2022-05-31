import importlib
from pathlib import Path
import struct as st
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import joblib
from keras.models import load_model
import numpy as np
from moleculekit.molecule import Molecule

from preprocessing.preprocess_xvg_file import get_processed_serial_and_label, make_list_extracted_md_serials, conv_atom_name_in_gropdb, add_chain_id_to_gropdb


def get_file_list(data_list):
    path = Path(data_list)
    with open(path, 'r') as f:
        lines = f.readlines()
    pdb_names, map_names, rmsf_xvg_names, gromacs_pdbs = [], [], [], []
    for l in lines:
        p, m, x, g = l.split()
        pdb_names.append(path.with_name(p))
        map_names.append(path.with_name(m))
        rmsf_xvg_names.append(path.with_name(x))
        gromacs_pdbs.append(path.with_name(g))
    return pdb_names, map_names, rmsf_xvg_names, gromacs_pdbs


def get_em_map(map_file):
    with open(map_file, 'rb') as f:
        header = f.read(1024)
        h_arr = np.asarray(st.unpack("256i", header))
        hf_arr = np.asarray(st.unpack("256f", header))
        sizex, sizey, stack = h_arr[0], h_arr[1], h_arr[2]
        nxstart = h_arr[4]
        grid_xsize = h_arr[7]
        cell_xsize = hf_arr[10]
        mapc = h_arr[16]  # map column [1 = x, 2 = y, 3 = z]

        data = sizex * sizey * stack
        body = f.read(data * 4)

        info = {'max_dist': stack,
                'resolution': cell_xsize / grid_xsize,
                'len': cell_xsize,
                'start_pos': nxstart,
                'map_column': mapc}

    return np.asarray(st.unpack(f"{data}f", body)).reshape((stack, sizey, sizex, 1)), info


def process_2dGMX_xvg(rmsf_xvg):
    with open(rmsf_xvg, 'r') as f:
        f = f.readlines()
        md_resid = [float(i.split()[0]) for i in f if i[0] != '#' and i[0] != '@']
        rmsf_val = [float(i.split()[1])*10 for i in f if i[0] != '#' and i[0] != '@']
    return np.asarray(md_resid), np.asarray(rmsf_val)


def standardize_int(em_arr, dtype=np.float32):
    em_arr_norm = (em_arr - np.mean(em_arr)) / np.std(em_arr)
    return np.where(em_arr_norm < 0, 0, em_arr_norm).astype(dtype)


def load_model_and_dataset(path_to_dataset, path_to_model=None, path_to_trained_model=None, train=True):
    dataset = joblib.load(path_to_dataset)
    if train:
        model_dot_path = os.path.splitext(path_to_model)[0].replace(os.path.sep, '.')
        import_model = importlib.import_module(model_dot_path)
        model = import_model.create_model(dataset.get("data"))
    else:
        model = load_model(str(path_to_trained_model))
    return model, dataset.get("data"), dataset.get("labels"), dataset.get("centers")


def get_voxel_with_label(pdb_file, rmsf_xvg, info, gromacs_pdb):
    coords_norm, labels = get_atom_lines_and_labels_from_pdb(pdb_file, rmsf_xvg, gromacs_pdb)
    coords_norm *= 1 / info['resolution']
    coords_norm = coords_norm.round() - info['start_pos']
    voxel = np.full((info['max_dist'], info['max_dist'], info['max_dist']), np.nan)
    for c, l in zip(coords_norm, labels):
        voxel[int(c[2])][int(c[1])][int(c[0])] = l
    return voxel


def get_atom_lines_and_labels_from_pdb(pdb_file, rmsf_xvg, gromacs_pdb):
    mol = Molecule(pdb_file)
    mol.filter('protein')  # comment out for bb filtered pdb
    mol_md = Molecule(gromacs_pdb)
    mol_md = conv_atom_name_in_gropdb(mol_md)
    mol_md = add_chain_id_to_gropdb(mol, mol_md)
    extracted_md_serials = make_list_extracted_md_serials(mol, mol_md)
    serid, rmsf_val = get_processed_serial_and_label(mol, rmsf_xvg, extracted_md_serials)
    label_list = []
    for i in mol.serial:
        label_list.append(rmsf_val[np.where(serid == i)])
    xyz = mol.get('coords')
    return xyz, label_list


def get_voxel(xyz, info):
    xyz_vox = xyz * (1 / info['resolution'])
    xyz_vox = xyz_vox.round() - info['start_pos']
    return xyz_vox


def standardize_values(log10rmsf_vals):
    avr_val = np.mean(log10rmsf_vals)
    std_val = np.std(log10rmsf_vals)
    return (log10rmsf_vals - avr_val) / std_val


def average_values(mol, log10rmsf):
    avr_log10rmsf = np.copy(log10rmsf)
    chids = mol.get('chain')
    _, chidx = np.unique(chids, return_index=True)
    chid_uniq = chids[np.sort(chidx)]
    rid_chid = [f'{r}{c}' for r, c in zip(mol.get('resid', sel='name CA'), mol.get('chain', sel='name CA'))]
    rid_chids = [f'{rs}{cs}' for rs, cs in zip(mol.get('resid'), mol.get('chain'))]
    for m, i in enumerate(rid_chid):
        identical_idx = [n for n, x in enumerate(rid_chids) if x == i]
        avr_val = np.mean(log10rmsf[identical_idx])
        avr_log10rmsf[identical_idx] = avr_val
    return avr_log10rmsf