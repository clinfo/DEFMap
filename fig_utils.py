import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


def make_mae_cost_plot(train_cost, valid_cost, train_mae, valid_mae, save_dir):
    plt.plot(train_cost, 'k-', label='Train Set Cost')
    plt.plot(valid_cost, 'r-', label='Validation Set Cost')
    plt.title('Mean Squared Error Loss per Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error Loss')
    plt.legend(loc='upper right')
    loss_path = os.path.join(save_dir, 'loss.png')
    plt.savefig(loss_path)
    print(f"[INFO] Save: {loss_path}")
    plt.clf()

    plt.plot(train_mae, 'k-', label='Train Set Mean Absolute Error')
    plt.plot(valid_mae, 'r-', label='Validation Set Mean Absolute Error')
    plt.title('Train and Validation Mean Absolute Error')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.legend(loc='upper right')
    mae_path = os.path.join(save_dir, 'mae.png')
    plt.savefig(mae_path)
    print(f"[INFO] Save: {mae_path}")
    plt.clf()


def correl_pred_plot(correct_vals, pred_vals, save_dir):
    pred_vals = pred_vals.flatten()
    corcoef = round(np.corrcoef(correct_vals, pred_vals)[0,1], 3)
    print(f'Corr. Coeff.: {corcoef}')
    X = [[x] for x in correct_vals]
    clf = linear_model.LinearRegression()
    clf.fit(X, pred_vals)
    a = clf.coef_
    b = clf.intercept_
    plt.scatter(correct_vals, pred_vals, marker='.', alpha=0.8, color='black')
    plt.plot(correct_vals, clf.predict(X), color='red')
    plt.text(2, -1, f'r = {corcoef}\ny = {str(round(a[0], 3))}x + {str(round(b, 3))}', size=10, bbox=dict(boxstyle='square', facecolor='white'))  # for log10 plot
    plt.title('Correlation between MD and 3D CNN')
    plt.xlabel('normalized log10(RMSF) derived from MD')         # for log10 plot
    plt.ylabel('normalized log10(RMSF) derived from 3D CNN')     # for log10 plot
    corr_path = os.path.join(save_dir, 'pred.png')
    plt.savefig(corr_path)
    print(f"[INFO] Save: {corr_path}")
    plt.clf()


def create_figures(history, result_dir):
    train_mae = history.history['mean_absolute_error']
    val_mae = history.history['val_mean_absolute_error']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    make_mae_cost_plot(train_loss, val_loss, train_mae, val_mae, result_dir)
