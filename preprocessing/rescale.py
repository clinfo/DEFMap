import argparse
import os
import struct as st
import sys
sys.path.append('../')
import subprocess

from moleculekit.molecule import Molecule
import numpy as np

from util import get_em_map


def get_parser():
    """
    Rescale voxel length and resolution in cryoEM map.
    Output: rescaled mrc file, rescaled pdb file.
    
    Returns:
        Argument parser
    """
    parser = argparse.ArgumentParser(
        description='Generation of rescaled cryoEM map.',
        usage=f"{os.path.basename(__file__)} -l <pdb_map.list file> -s <desired scale (A/pixel)> "
              f"-r <desired resolution (A)>"
    )
    parser.add_argument(
        "-l", "--data_list", required=False, type=str,
        help="a list of multiple pdb and the corresponding mrc/map files."
    )
    parser.add_argument(
        "-s", "--scaled_length", required=True, type=float,
        help="Voxel length (angstrom/pixel) in rescaled cryoEM map."
    )
    parser.add_argument(
        "-r", "--resolution", required=True, type=float,
        help="Resolution (angstrom) in rescaled cryoEM map."
    )
    parser.add_argument(
        "-m", "--map_file", required=False, type=str,
        help="path to cryoEM map file"
    )
    return parser.parse_args()


def get_file_list(data_list):
    with open(data_list, 'r') as f:
        lines = f.readlines()
    pdb_names, map_names = [], []
    for l in lines:
        p, m = l.split()
        pdb_names.append(p)
        map_names.append(m)
    return pdb_names, map_names


def conv_map_order(map_file):
    with open(map_file, 'rb') as f:
        header = f.read(1024)
        h_arr = np.asarray(st.unpack("256i", header))
        sizex, sizey, stack = h_arr[0], h_arr[1], h_arr[2]
        # change em_int order in header (3, 2, 1) -> (1, 2, 3)
        h_arr[16] = 1
        h_arr[18] = 3
        data = sizex * sizey * stack
        body = f.read(data * 4)
        em_array = np.asarray(st.unpack(f"{data}f", body)).reshape((sizex, sizey, stack, 1))
        new_em_array = np.zeros((stack, sizey, sizex, 1))
        idx_list = ([(x, y, z) for x in range(sizex)
                     for y in range(sizey)
                     for z in range(stack)])
        for idx in idx_list:
            new_em_array[idx[2], idx[1], idx[0]] = em_array[idx[0], idx[1], idx[2]]
    with open('tmp.mrc', 'wb') as fb:
        for h in h_arr:
            fb.write(st.pack("i", h))
        for i in new_em_array.ravel():
            fb.write(st.pack("f", i))


def main():
    args = get_parser()
    if args.map_file:
        map_files = [args.map_file]
    else:
        pdb_files, map_files = get_file_list(args.data_list)
    for i in range(len(map_files)):
        map_file = map_files[i] if args.map_file else os.path.join(os.path.dirname(args.data_list), map_files[i])
        print(f"[INFO] load {map_file}")
        em_arr, info = get_em_map(map_file)
        scale_val = info['resolution']/args.scaled_length
        freq = 1/args.resolution
        fname = f"{os.path.splitext(os.path.basename(map_files[i]))[0]}_{args.resolution}A_rescaled.mrc"
        output_dir = os.path.dirname(args.map_file) if args.map_file else os.path.dirname(args.data_list)
        output = os.path.join(output_dir, fname)
        if info['map_column'] == 3:
            conv_map_order(map_file)
            cmd1 = f"e2proc3d.py tmp.mrc {output} " \
                   f"--clip={info['max_dist']},{info['max_dist']},{info['max_dist']} " \
                   f"--scale={scale_val} " \
                   f"--process=filter.lowpass.gauss:cutoff_freq={freq}"
            subprocess.run(cmd1.split())
            os.system('rm -rf tmp.mrc')
        else:
            cmd1 = f"e2proc3d.py {map_file} {output} " \
                   f"--clip={info['max_dist']},{info['max_dist']},{info['max_dist']} " \
                   f"--scale={scale_val} " \
                   f"--process=filter.lowpass.gauss:cutoff_freq={freq}"
            subprocess.run(cmd1.split())
        print(f"[INFO] save at {output}")
        if args.map_file:
            continue

        pdb_file = os.path.join(os.path.dirname(args.data_list), pdb_files[i])
        print(f"[INFO] load {pdb_file}")
        mol = Molecule(pdb_file)
        mol.filter('protein')
        xyz = mol.get('coords')
        delta = (info['max_dist']/2 + info['start_pos']) * (info['resolution'] - args.scaled_length)
        xyz -= delta
        mol.set('coords', xyz)
        output_pdb = f'{os.path.splitext(pdb_file)[0]}_rescaled.pdb'
        mol.write(output_pdb)
        print(f"[INFO] save at {output_pdb}")


if __name__ == "__main__":
    main()
