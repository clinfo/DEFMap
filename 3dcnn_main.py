import argparse
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import joblib
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping

from fig_utils import correl_pred_plot, create_figures
from util import load_model_and_dataset


def get_parser():
    parser = argparse.ArgumentParser(
        description='Prediction for SSE in low resolution EM map with 3D CNN.',
        usage=f"{os.path.basename(__file__)} -d <path to dataset> -m <model name>"
    )
    parser.add_argument(
        'mode', type=str, choices=['train', 'infer', 'train_infer'],
        help="choose [train, infer, train_infer]"
    )
    parser.add_argument(
        '-d', '--train_dataset', type=str,
        help="path to dataset file for training"
    )
    parser.add_argument(
        "-t", "--test_dataset", type=str,
        help="path to dataset file for inference"
    )
    parser.add_argument(
        "-o", "--model_output", type=str, default="model/model.h5",
        help="model name for save"
    )
    parser.add_argument(
        "-p", "--prediction_output", type=str, default="prediction.jbl",
        help="path to prediction result file"
    )
    parser.add_argument(
        "-m", "--model", type=str, default="model/sample_model1.py",
        help="path to a model definition file"
    )
    parser.add_argument(
        "-r", "--result_dir", type=str, default="result",
        help="path to a result directory"
    )
    parser.add_argument(
        "-g", "--gpu", type=str,
        help="constrain gpu. (e.g. 0,1)"
    )
    return parser.parse_args()


def infer(args):
    print("[INFO] Inference")
    trained_model_path = Path(args.model_output)
    os.makedirs(trained_model_path.parent, exist_ok=True)
    model, data, labels, centers = load_model_and_dataset(args.test_dataset, path_to_trained_model=trained_model_path,
                                                          train=False)
    log_val = model.predict(data)
    obj = {c: v for c, v in zip(centers, log_val)}
    joblib.dump(obj, args.prediction_output)
    print(f"[INFO] Save: {args.prediction_output}")
    if labels is not None:  # evaluation
        score = model.evaluate(data, labels, verbose=0)
        print(f"Test loss: {score[0]}\n")
        correl_pred_plot(labels, log_val, args.result_dir)
    tf.keras.backend.clear_session()


def train(args):
    print("[INFO] Training")
    trained_model_path = Path(args.model_output)
    os.makedirs(trained_model_path.parent, exist_ok=True)
    model, data, labels, _ = load_model_and_dataset(args.train_dataset, path_to_model=args.model, train=True)
    if args.mode == "train":
        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2,
                                                                            random_state=1234)
        train_data, valid_data, train_labels, valid_labels = train_test_split(train_data, train_labels, test_size=0.25,
                                                                              random_state=1234)
    else:  # args.mode == "train_infer":
        train_data, valid_data, train_labels, valid_labels = train_test_split(data, labels, test_size=0.25,
                                                                              random_state=1234)
    #
    os.makedirs(args.result_dir, exist_ok=True)
    # Setting for callbacks
    save_path = os.path.join(args.result_dir, "train.log")
    logger = CSVLogger(save_path)
    print(f"[INFO] Save: {save_path}")
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    callbacks = [logger, early_stopping]
    #
    history = model.fit(train_data, train_labels, validation_data=(valid_data, valid_labels), batch_size=128,
                        epochs=100, verbose=1, shuffle=True, callbacks=callbacks)
    if args.mode == "train":
        score = model.evaluate(test_data, test_labels, verbose=0)
        print(f"Test RMSE: {score[0]}\n")
    print(f"[INFO] Save: {trained_model_path}")
    model.save(str(trained_model_path))
    create_figures(history, args.result_dir)
    tf.keras.backend.clear_session()


def main():
    args = get_parser()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.mode == "train":
        train(args)
    elif args.mode == "infer":
        infer(args)
    elif args.mode == "train_infer":
        train(args)
        infer(args)
    print("[INFO] Done")


if __name__ == "__main__":
    main()
