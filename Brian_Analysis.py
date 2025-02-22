import os
import json
import numpy as np
import matplotlib.pyplot as plt

from osl_dynamics.inference.metrics import twopair_riemannian_distance
from osl_dynamics.inference.modes import hungarian_pair
from osl_dynamics.utils.plotting import plot_mode_pairing
if __name__ == '__main__':
    save_dir_1 = './results_final/real/ICA_50_UKB_first_scan/hmm_state_10/'
    save_dir_2 = './results_final/real/ICA_50_UKB_second_scan/hmm_state_10/'
    plot_dir = './Brian_Analysis_results/'

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Step 1: Evaluate the split-half reproducibility within each model
    # First scan reproducibility
    split_half_metric = []
    for i in range(1,6):
        cov_1 = np.load(f'{save_dir_1}/split{i}/partition_1/inf_params/covs.npy')
        cov_2 = np.load(f'{save_dir_1}/split{i}/partition_1/inf_params/covs.npy')
        riem = twopair_riemannian_distance(cov_1, cov_2)
        indice, riem_reorder = hungarian_pair(riem, distance=True)
        plot_mode_pairing(riem_reorder, indice, x_label='2nd half states', y_label='1st half states',
                          filename=f'{plot_dir}/split_half_first_scan_{i}.jpg')
        split_half_metric.append(float(np.mean(np.diagonal(riem_reorder))))
    with open(f"{plot_dir}/split_half_first_scan.json", "w") as f:
        json.dump(split_half_metric, f)

    # Second scan reproducibility
    split_half_metric = []
    for i in range(1, 6):
        cov_1 = np.load(f'{save_dir_2}/split{i}/partition_1/inf_params/covs.npy')
        cov_2 = np.load(f'{save_dir_2}/split{i}/partition_1/inf_params/covs.npy')
        riem = twopair_riemannian_distance(cov_1, cov_2)
        indice, riem_reorder = hungarian_pair(riem, distance=True)
        plot_mode_pairing(riem_reorder, indice, x_label='2nd half states', y_label='1st half states',
                            filename=f'{plot_dir}/split_half_second_scan{i}.jpg')
        split_half_metric.append(float(np.mean(np.diagonal(riem_reorder))))
    with open(f"{plot_dir}/split_half_second_scan.json", "w") as f:
        json.dump(split_half_metric, f)