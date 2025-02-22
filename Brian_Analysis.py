import os
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt

from osl_dynamics.config_api.wrappers import load_data
from osl_dynamics.models import load


from osl_dynamics.inference.metrics import twopair_riemannian_distance
from osl_dynamics.inference.modes import hungarian_pair
from osl_dynamics.utils.plotting import plot_mode_pairing
if __name__ == '__main__':
    save_dir_1 = './results_final/real/ICA_50_UKB_first_scan/hmm_state_10/'
    save_dir_2 = './results_final/real/ICA_50_UKB_second_scan/hmm_state_10/'
    plot_dir = './Brian_Analysis_results/'

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    '''
    # Step 1: Evaluate the split-half reproducibility within each model
    # First scan reproducibility
    split_half_metric = []
    for i in range(1,6):
        cov_1 = np.load(f'{save_dir_1}/split_{i}/partition_1/inf_params/covs.npy')
        cov_2 = np.load(f'{save_dir_1}/split_{i}/partition_2/inf_params/covs.npy')
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
        cov_1 = np.load(f'{save_dir_2}/split_{i}/partition_1/inf_params/covs.npy')
        cov_2 = np.load(f'{save_dir_2}/split_{i}/partition_2/inf_params/covs.npy')
        riem = twopair_riemannian_distance(cov_1, cov_2)
        indice, riem_reorder = hungarian_pair(riem, distance=True)
        plot_mode_pairing(riem_reorder, indice, x_label='2nd half states', y_label='1st half states',
                            filename=f'{plot_dir}/split_half_second_scan{i}.jpg')
        split_half_metric.append(float(np.mean(np.diagonal(riem_reorder))))
    with open(f"{plot_dir}/split_half_second_scan.json", "w") as f:
        json.dump(split_half_metric, f)

    # Step 2: Scan_1, Scan_2 Reproducibility
    scan_metric = []
    for i in range(1,4):
        for j in range(1,4):
            cov_1 = np.load(f'{save_dir_1}/repeat_{i}/inf_params/covs.npy')
            cov_2 = np.load(f'{save_dir_2}/repeat_{j}/inf_params/covs.npy')
            riem = twopair_riemannian_distance(cov_1, cov_2)
            indice, riem_reorder = hungarian_pair(riem, distance=True)
            plot_mode_pairing(riem_reorder, indice, x_label='2nd half states', y_label='1st half states',
                              filename=f'{plot_dir}/scan_reproducibility_{i}_{j}.jpg')
            scan_metric.append(float(np.mean(np.diagonal(riem_reorder))))

            # Save indices for later use
            with open(f"{plot_dir}/state_matching_indices_{i}_{j}.json", "w") as f:
                json.dump(indice, f)
    with open(f"{plot_dir}/scan_reproducibility.json", "w") as f:
        json.dump(scan_metric, f)
    
    # Step 3: Dual estimation for each subject
    # Scan 1
    with open(f"{save_dir_1}/repeat_1/prepared_config.yaml", "r") as file:
        config_1 = yaml.safe_load(file)
    load_data_kwargs_1 = config_1['load_data']
    data_1 = load_data(**load_data_kwargs_1)

    for i in range(1,4):
        model = load(f'{save_dir_1}/repeat_{i}/model/')
        _, covs = model.dual_estimation(data_1)
        np.save(f'{plot_dir}/first_scan_covs_{i}.npy',covs)

    # Scan 2
    with open(f"{save_dir_2}/repeat_1/prepared_config.yaml", "r") as file:
        config_2 = yaml.safe_load(file)
    load_data_kwargs_2 = config_2['load_data']
    data_2 = load_data(**load_data_kwargs_2)

    for i in range(1,4):
        model = load(f'{save_dir_2}/repeat_{i}/model/')
        _, covs = model.dual_estimation(data_2)
        np.save(f'{plot_dir}/second_scan_covs_{i}.npy',covs)
    '''


    # Step 4: Create subject-specific feature vectors using matched states
    def create_feature_vectors(covs, indices):
        N_subjects, N_states, N_channels, _ = covs.shape
        upper_tri_indices = np.triu_indices(N_channels)  # Get upper triangular indices
        feature_vectors = np.zeros((N_subjects, N_states * len(upper_tri_indices[0])))

        for subj in range(N_subjects):
            feature_vectors[subj] = np.concatenate(
                [covs[subj, indices[state]][upper_tri_indices] for state in range(N_states)])

        return feature_vectors

    '''
    # Define a very simple test case:
    covs = np.array([
        [
        [[1.0,0.5],
        [0.5,1.0]],
        [[1.0,-0.5],
        [-0.5,1.0]]
        ],
        [
        [[1.0, 0.2],
        [0.2, 1.0]],
        [[1.0, -0.2],
        [-0.2, 1.0]]
        ]
    ])
    indices = [1,0]
    print('Real output is:',create_feature_vectors(covs,indices))
    print('The expected output is:')
    print(np.array([
        [1.0,1.0,-0.5,1.0,1.0,0.5],
        [1.0,1.0,-0.2,1.0,1.0,0.2]
    ]))
    
    with open(f"{save_dir_1}/repeat_1/prepared_config.yaml", "r") as file:
        config_1 = yaml.safe_load(file)
    load_data_kwargs_1 = config_1['load_data']
    data_1 = load_data(**load_data_kwargs_1)

    # Scan 2
    with open(f"{save_dir_2}/repeat_1/prepared_config.yaml", "r") as file:
        config_2 = yaml.safe_load(file)
    load_data_kwargs_2 = config_2['load_data']
    data_2 = load_data(**load_data_kwargs_2)

    sFC_1, sFC_2 = [],[]
    for ts in data_1.time_series():
        sFC_1.append(np.corrcoef(ts, rowvar=False))

    for ts in data_2.time_series():
        sFC_2.append(np.corrcoef(ts,rowvar=False))
    sFC_1 = np.array(sFC_1)
    sFC_2 = np.array(sFC_2)
    np.save(f"{plot_dir}/static_first_scan.npy",sFC_1)
    np.save(f"{plot_dir}/static_second_scan.npy",sFC_2)
    '''
    sFC_1 = np.load(f'{plot_dir}/static_first_scan.npy')
    sFC_2 = np.load(f'{plot_dir}/static_second_scan.npy')
    upper_tri_indices = np.triu_indices(50, k=0)  # k=0 includes the diagonal
    sFC_1_flattened = sFC_1[:400, upper_tri_indices[0], upper_tri_indices[1]]
    sFC_2_flattened = sFC_2[:400, upper_tri_indices[0], upper_tri_indices[1]]

    N_subjects = len(sFC_1)

    # Compute Between-Session Similarity Matrix using Pearson correlation
    similarity_matrix = np.corrcoef(sFC_1_flattened, sFC_2_flattened)[:N_subjects, N_subjects:]

    # Evaluate Subject Label Prediction Accuracy
    correct_matches = np.argmax(similarity_matrix, axis=1) == np.arange(N_subjects)
    prediction_accuracy = np.mean(correct_matches)
    print('sFC prediction accuracy:', prediction_accuracy)

    # Compute Top-5 Accuracy
    top5_predictions = np.argsort(similarity_matrix, axis=1)[:, -5:]
    top5_correct_matches = [i in top5_predictions[i] for i in range(N_subjects)]
    top5_accuracy = np.mean(top5_correct_matches)
    print('sFC top 5 accuracy:',top5_accuracy)

    # Calculate static functional connectivity
    # Step 5 & 6: Compute similarity matrix and evaluate prediction accuracy
    for i in range(1, 4):
        for j in range(1, 4):
            # Load covariance matrices
            session1_covs = np.load(f'{plot_dir}/first_scan_covs_{i}.npy')
            session2_covs = np.load(f'{plot_dir}/second_scan_covs_{j}.npy')

            N_subjects = len(session1_covs)

            # Load corresponding state-matching indices
            with open(f"{plot_dir}/state_matching_indices_{i}_{j}.json", "r") as f:
                indices = json.load(f)

            # Generate feature vectors for both sessions using matched states
            session1_features = create_feature_vectors(session1_covs, indices['row'])
            session2_features = create_feature_vectors(session2_covs, indices['col'])

            # Compute Between-Session Similarity Matrix using Pearson correlation
            similarity_matrix = np.corrcoef(session1_features, session2_features)[:N_subjects, N_subjects:]

            # Evaluate Subject Label Prediction Accuracy
            correct_matches = np.argmax(similarity_matrix, axis=1) == np.arange(N_subjects)
            prediction_accuracy = np.mean(correct_matches)

            # Save results
            np.save(f'{plot_dir}/between_session_similarity_{i}_{j}.npy', similarity_matrix)
            with open(f"{plot_dir}/subject_label_prediction_accuracy_{i}_{j}.txt", "w") as f:
                f.write(f"Subject Label Prediction Accuracy: {prediction_accuracy:.4f}\n")

            print(f"Iteration ({i},{j}) - Subject Label Prediction Accuracy: {prediction_accuracy:.4f}")