import argparse
from math import isnan
import os
import sys
sys.path.append('../')
from tqdm import tqdm

import joblib
import numpy as np
from scipy.ndimage import rotate

from util import get_em_map, standardize_int, get_file_list, get_voxel_with_label


def get_parser():
    """
    In the input data list file, space/tab-separated pdb, map, rmsf.xvg file names are indicated.
    The order, pdb_file_name (1), map_file_name (2) and rmsf_xvg_name (3), has to be fixed.
    e.g.) 004_6mix.pdb 004_emd_9132_10A.mrc 004_rmsf.xvg

    prep_dataset.py is used for labels without residual average of each atom RMSF value.
    Pre-processing of raw rmsf.xvg file by preprocess_xvg_file.py are required.

    Returns:
        Argument parser
    """
    parser = argparse.ArgumentParser(
        description='Dataset preparation for 3D-CNN to predict log10(RMSF) from low resolution.',
        usage=f"{os.path.basename(__file__)} -l <pdb_map.list file> -o <output filename> (-a) (-v <voxel_range>) ((-p) (-m))"
    )
    parser.add_argument(
        "-l", "--data_list", type=str,
        help="path to a list file that contains multiple pdb file name and the corresponding mrc/map and rmsf.xvg file name."
    )
    parser.add_argument(
        '-o', '--output', type=str, default="data/dataset.jbl",
        help="output file name"
    )
    parser.add_argument(
        "-a", "--data_augment", action="store_true",
        help="data augmentation"
    )
    parser.add_argument(
        "-v", "--voxel_range", type=int, default=10,
        help="voxel range. specify even number"
    )
    parser.add_argument(
        "-p", "--prediction", action="store_true",
        help="create a dataset for prediction"
    )
    parser.add_argument(
        '-t', '--threshold', action='store', default=None, type=float,
        help='threshold to drop sub-voxels with a standardized intensity less than the threshold'
    )
    parser.add_argument(
        '-s', '--save_memory', action='store_true',
        help='switch numpy.dtype from numpy.float32 to numpy.float16'
    )
    parser.add_argument(
        "-m", "--map_file", type=str,
        help="path to a map file."
    )
    return parser.parse_args()


def create_dataset(em_arr_norm, pdb_file, rmsf_xvg, info, args, gromacs_pdb):
    data, labels, centers = [], [], []
    voxel = get_voxel_with_label(pdb_file, rmsf_xvg, info, gromacs_pdb)
    hrange = int(args.voxel_range / 2)
    center_list = np.asarray([(x, y, z) for x in range(hrange, info['max_dist'] - hrange + 1)
                              for y in range(hrange, info['max_dist'] - hrange + 1)
                              for z in range(hrange, info['max_dist'] - hrange + 1)])
    for center in center_list:
        if isnan(voxel[center[0]][center[1]][center[2]]):
            continue
        sub_voxel = em_arr_norm[center[0] - hrange:center[0] + hrange,
                                center[1] - hrange:center[1] + hrange,
                                center[2] - hrange:center[2] + hrange]
        label = voxel[center[0]][center[1]][center[2]]
        center = ','.join(list(map(str, center.tolist())))
        if args.data_augment:
            aug_sub_voxel, aug_label, aug_center = generate_rotate_voxels(sub_voxel, label, center)
            data.extend(aug_sub_voxel)
            labels.extend(aug_label)
            centers.extend(aug_center)
        else:
            data.append(sub_voxel)
            labels.append(label)
            centers.append(center)
    return np.array(data, dtype=np.float32), np.log10(labels), centers


def create_dataset_for_prediction(em_arr_norm, info, args):
    data, centers = [], []
    hrange = int(args.voxel_range / 2)
    center_list = np.asarray([(x, y, z) for x in range(hrange, info['max_dist'] - hrange + 1)
                              for y in range(hrange, info['max_dist'] - hrange + 1)
                              for z in range(hrange, info['max_dist'] - hrange + 1)], dtype=np.int16)
    set_threshold = False if args.threshold is None else True
    for center in center_list:
        label = em_arr_norm[center[0]][center[1]][center[2]][0]
        if isnan(label):
            continue
        if set_threshold:
            if not label > args.threshold:
                continue
        sub_voxel = em_arr_norm[center[0] - hrange:center[0] + hrange,
                                center[1] - hrange:center[1] + hrange,
                                center[2] - hrange:center[2] + hrange]
        center = ','.join(list(map(str, center.tolist())))
        data.append(sub_voxel)
        centers.append(center)
    return np.array(data), centers


def generate_rotate_voxels(voxel, label, center):
    sub_voxels = [voxel]
    # x-y plane rotation
    for angle in (90, 180, 270):
        sub_voxel = rotate(voxel, axes=(0, 1), angle=angle, cval=0.0, reshape=False)
        sub_voxels.append(sub_voxel)
    # x-z plane rotation
    for angle in (90, 180, 270):
        sub_voxel = rotate(voxel, axes=(0, 2), angle=angle, cval=0.0, reshape=False)
        sub_voxels.append(sub_voxel)
    # y-z plane rotation
    for angle in (90, 180, 270):
        sub_voxel = rotate(voxel, axes=(1, 2), angle=angle, cval=0.0, reshape=False)
        sub_voxels.append(sub_voxel)
    labels = [label for _ in range(len(sub_voxels))]
    centers = [center for _ in range(len(sub_voxels))]
    return sub_voxels, labels, centers


def save_dataset(data, labels, centers, output_filename):
    obj = {'data': data,
           'labels': labels,
           'centers': centers}
    joblib.dump(obj, output_filename)
    print(f"[INFO] Save train data set: {output_filename}")


def save_dataset_for_prediction(data, centers, output_filename):
    obj = {"data": data,
           "centers": centers}
    joblib.dump(obj, output_filename)
    print(f"[INFO] Save prediction data set: {output_filename}")


def main():
    args = get_parser()
    dtype = np.float16 if args.save_memory else np.float32
    if args.prediction:
        em_arr, info = get_em_map(args.map_file)
        em_arr_norm = standardize_int(em_arr, dtype=dtype)
        data, centers = create_dataset_for_prediction(em_arr_norm, info, args)
        save_dataset_for_prediction(data, centers, args.output)
        print("[INFO] Done")
        sys.exit(0)
    pdb_file, map_file, rmsf_xvg, gromacs_pdbs = get_file_list(args.data_list)
    data_list, labels_list, centers_list = [], [], []
    for i in tqdm(range(len(pdb_file))):
        em_arr, info = get_em_map(map_file[i])
        em_arr_norm = standardize_int(em_arr, dtype=dtype)
        data, labels, centers = create_dataset(em_arr_norm, str(pdb_file[i]), str(rmsf_xvg[i]), info, args, str(gromacs_pdbs[i]))
        data_list.extend(data)
        labels_list.extend(labels)
        centers_list.extend(centers)
    save_dataset(np.array(data_list, dtype=dtype), np.array(labels_list, dtype=dtype), centers_list, args.output)
    print("[INFO] Done")


if __name__ == "__main__":
    main()
