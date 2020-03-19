import argparse
import os
import struct as st

import joblib
import numpy as np
from moleculekit.molecule import Molecule


def get_parser():
    parser = argparse.ArgumentParser(
        description='',
        usage=''
    )
    parser.add_argument(
        '-m', '--map_file', action='store', default=None, required=True, type=str,
        help='help'
    )
    parser.add_argument(
        '-p', '--prediction_output', action='store', default=None, required=True, type=str,
        help='help'
    )
    parser.add_argument(
        '-t', '--threshold', action='store', default=None, required=True, type=float,
        help='help'
    )
    return parser.parse_args()


def get_em_map(map_file):
    with open(map_file, 'rb') as f:
        header = f.read(1024)
        h_arr = np.asarray(st.unpack("256i", header))
        sizex, sizey, stack = h_arr[0], h_arr[1], h_arr[2]
        data = sizex * sizey * stack
        body = f.read(data * 4)
    return np.asarray(st.unpack(f"{data}f", body)).reshape((stack, sizey, sizex))


def main():
    args = get_parser()
    print(f"[INFO] load {args.prediction_output}")
    data = joblib.load(args.prediction_output)
    print(f"[INFO] load {args.map_file}")
    em_map = get_em_map(args.map_file)
    em_map_coords = np.array(np.where(em_map > args.threshold)).T
    map_keys = {','.join(map(str, coord)) for coord in em_map_coords}

    filtered_dict = dict(filter(lambda x: x[0] in map_keys, data.items()))
    beta = np.squeeze(np.array(list(filtered_dict.values())))
    coords = [x.split(',') for x in list(filtered_dict.keys())]
    coords = np.array([list(map(int, i)) for i in coords])

    mol = Molecule().empty(numAtoms=len(filtered_dict.keys()))
    mol.set('record', 'ATOM')
    mol.set('resname', 'MG')
    mol.set('element', 'MG')
    mol.box = np.array([[0.], [0.], [0.]], dtype=np.float32)
    mol.coords =  coords[:, :, np.newaxis].astype(np.float32)
    mol.set('beta', beta)
    filename = os.path.basename(args.map_file)
    filename = f"{os.path.splitext(filename)[0]}.pdb"
    print(f"[INFO] save to {filename}")
    mol.write(filename)


if __name__ == "__main__":
    main()

