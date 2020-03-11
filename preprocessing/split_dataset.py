import argparse
from operator import itemgetter
import os
from sklearn.model_selection import KFold


def get_parser():
    parser = argparse.ArgumentParser(
        description='description',
        usage='usage'
    )
    parser.add_argument(
        '-i', '--input', action='store', default=None, required=True,
        help='help'
    )
    return parser.parse_args()


def main():
    args = get_parser()
    with open(args.input, 'r') as f:
        lines = [l.strip("\n") for l in f.readlines()]
    kf = KFold(n_splits=5, shuffle=True, random_state=1234)
    for i, (train_idxs, test_idxs) in enumerate(kf.split(lines)):
        train_out = '\n'.join(itemgetter(*train_idxs)(lines))
        test_out = lines[test_idxs[0]] if len(test_idxs) == 1 else '\n'.join(itemgetter(*test_idxs)(lines))
        train_out_path = os.path.join(os.path.dirname(args.input), f"train_fold_{i}.txt")
        test_out_path = os.path.join(os.path.dirname(args.input), f"test_fold_{i}.txt")
        print(f"[INFO] {train_out_path}")
        print(f"[INFO] {test_out_path}")
        with open(train_out_path, "w") as f_train, open(test_out_path, "w") as f_test:
            f_train.write(train_out)
            f_test.write(test_out)


if __name__ == "__main__":
    main()
