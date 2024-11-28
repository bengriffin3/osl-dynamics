import numpy as np
import numpy.testing as npt
from osl_dynamics.config_api.wrappers import load_data

def test_load_data():
    import os
    import json
    save_dir = './test_temp/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    vector = np.array([-1.5 ** 0.5, 0, 1.5 ** 0.5])
    input_1 = np.array([vector, vector + 10.0]).T
    input_2 = np.array([vector * 0.5 + 1., vector * 100]).T
    np.savetxt(f'{save_dir}10001.txt', input_1)
    np.savetxt(f'{save_dir}10002.txt', input_2)
    prepare = {'standardize':{}}

    data = load_data(inputs=save_dir,prepare=prepare)
    npt.assert_almost_equal(data.arrays[0],np.array([vector,vector]).T)
    npt.assert_almost_equal(data.arrays[1],np.array([vector,vector]).T)
    # Remove the directory after testing
    from shutil import rmtree
    rmtree(save_dir)


def test_infer_spatial_hmm():
    import os
    import json
    import pickle
    import shutil
    import yaml
    from osl_dynamics.evaluate.cross_validation import BCV

    save_dir = './test_infer_spatial/'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # Define a very simple test case
    n_samples = 3
    n_channels = 3
    n_states = 3
    row_train = [1, 2]
    column_X = [1]
    column_Y = [0, 2]
    row_test = [0]

    indices_dir = f"{save_dir}/indices.json"
    with open(indices_dir, "w") as json_file:
        json.dump({
            'row_train':row_train,
            'row_test':row_test,
            'column_X':column_X,
            'column_Y':column_Y
        }, json_file, indent=4)

    # Construct the data
    def generate_obs(cov, mean=None, n_timepoints=100):
        if mean is None:
            mean = np.zeros(len(cov))
        return np.random.multivariate_normal(mean, cov, n_timepoints)

    # Define the covariance matrices of state 1,2 in both splits
    cors_Y = [-0.5, 0.0, 0.5]
    covs_Y = [np.array([[1.0, cor], [cor, 1.0]]) for cor in cors_Y]

    means_X = [1.0, 2.0, 3.0]
    vars_X = [0.5, 1.0, 2.0]

    n_timepoints = 100

    # save these files
    data_dir = f'{save_dir}data/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    hidden_states = []

    for i in range(0, 2):
        # Build up the hidden variable
        hv_temp = np.zeros((n_timepoints * 2, n_states))
        hv_temp[:, i] = np.array([0.6] * n_timepoints + [0.4] * n_timepoints)
        hv_temp[:, i + 1] = np.array([0.4] * n_timepoints + [0.6] * n_timepoints)
        hidden_states.append(np.tile(hv_temp, (1500, 1)))

        obs = []
        for j in range(1500):
            observations_Y = [generate_obs(covs_Y[i], n_timepoints=n_timepoints),
                              generate_obs(covs_Y[i + 1], n_timepoints=n_timepoints)]
            observations_X = [generate_obs([[vars_X[i]]], [means_X[i]], n_timepoints=n_timepoints),
                              generate_obs([[vars_X[i + 1]]], [means_X[i + 1]], n_timepoints=n_timepoints)]
            observations = np.concatenate(
                [np.hstack((Y[:, :1], X, Y[:, 1:])) for X, Y in zip(observations_X, observations_Y)], axis=0)
            obs.append(observations)

        obs = np.concatenate(obs, axis=0)
        np.save(f"{data_dir}{10002 + i}.npy", obs)

    with open(f'{data_dir}alp.pkl', "wb") as file:
        pickle.dump(hidden_states, file)
    # Genetate irrelevant dataset
    np.save(f"{data_dir}10001.npy", generate_obs(np.eye(3) * 100, n_timepoints=300000))

    config = f"""
            load_data:
                inputs: {data_dir}
                prepare:
                    select:
                        timepoints:
                            - 0
                            - 300000
                kwargs:
                  load_memmaps: True
            save_dir: {save_dir}
            indices: {indices_dir}
            mode: bcv_1
            model:
              hmm:
                config_kwargs:
                  n_states: {n_states}
                  learn_means: True
                  learn_covariances: True
                  learning_rate: 0.01
                  n_epochs: 3
                  sequence_length: 600
            """

    config = yaml.safe_load(config)
    bcv = BCV(config)
    result = bcv.infer_spatial(row_train, column_X, f'{data_dir}alp.pkl',method='sample')

    result_means = np.load(result['means'])
    result_covs = np.load(result['covs'])
    npt.assert_allclose(means_X, result_means, rtol=1e-2, atol=1e-2)
    npt.assert_allclose(vars_X, result_covs, rtol=1e-2, atol=1e-2)



def test_train_swc():
    import os
    import pickle
    import shutil
    import yaml
    from osl_dynamics.config_api.wrappers import train_swc
    from osl_dynamics.config_api.pipeline import run_pipeline_from_file

    save_dir = './test_train_swc/'
    data_dir = f'{save_dir}/data/'
    output_dir = f'{save_dir}/result/'

    if os.path.isdir(save_dir):
        # Remove the directory
        shutil.rmtree(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_1 = np.array([[1.0, -1.0],
                       [0.0, 0.0],
                       [-1.0, 1.0],
                       [2.0, -2.0],
                       [-2.0, 2.0],
                       [0.0, 0.0],
                       [10.0, 10.0],
                       [-10.0, -10.0],
                       ])
    data_2 = np.array([[2.0, -2.0],
                       [-2.0, 2.0],
                       [0.0, 0.0],
                       [10.0, 10.0],
                       [-10.0, -10.0],
                       [0.0, 3.0],
                       [-3.0, 0.0],
                       [3.0, -3.0]
                       ])
    np.save(f'{data_dir}/10001.npy', data_1)
    np.save(f'{data_dir}/10002.npy', data_2)

    config = f"""
            load_data:
                inputs: {data_dir}
            train_swc:
                config_kwargs:
                    n_channels: 2
                    n_states: 3
                    learn_means: False
                    learn_covariances: True
                    window_length: 5
                    window_offset: 3
                """
    with open(f'{output_dir}train_config.yaml', "w") as file:
        yaml.safe_dump(yaml.safe_load(config), file, default_flow_style=False)
    run_pipeline_from_file(f'{output_dir}train_config.yaml', output_dir)

    # After running, check the files in the directory
    inf_params_dir = f'{output_dir}/inf_params/'
    means = np.load(f'{inf_params_dir}/means.npy')
    covs = np.load(f'{inf_params_dir}/covs.npy')
    with open(f'{inf_params_dir}/alp.pkl','rb') as file:
        labels = pickle.load(file)

    covs_answer = np.array([[[2.5, -2.5], [-2.5, 2.5]],
                            [[52., 48.], [48., 52.]],
                            [[54.5, 47.75], [47.75, 54.5]]])
    labels_answer = [np.array([0, 1]), np.array([1, 2])]

    # Reorder and covs and update the labels accordingly
    order = np.argsort(covs[:, 0, 0])
    reordered_covs = covs[order]
    label_mapping = {old: new for new, old in enumerate(order)}
    reordered_labels = [np.array([label_mapping[label] for label in session_labels]) for session_labels in labels]

    npt.assert_array_equal(means,np.zeros((3,2)))
    npt.assert_array_equal(covs_answer, reordered_covs)
    npt.assert_array_equal(labels_answer, reordered_labels)

def test_swc_infer_spatial():
    import os
    import pickle
    import shutil
    import yaml
    from osl_dynamics.config_api.pipeline import run_pipeline_from_file

    save_dir = './test_swc_infer_spatial/'
    data_dir = f'{save_dir}/data/'
    output_dir = f'{save_dir}/result/'

    if os.path.isdir(save_dir):
        # Remove the directory
        shutil.rmtree(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_1 = np.array([[1.0, -1.0],
                       [0.0, 0.0],
                       [-1.0, 1.0],
                       [2.0, -2.0],
                       [-2.0, 2.0],
                       [0.0, 0.0],
                       [10.0, 10.0],
                       [-10.0, -10.0],
                       ])
    data_2 = np.array([[2.0, -2.0],
                       [-2.0, 2.0],
                       [0.0, 0.0],
                       [10.0, 10.0],
                       [-10.0, -10.0],
                       [0.0, 3.0],
                       [-3.0, 0.0],
                       [3.0, -3.0]
                       ])
    np.save(f'{data_dir}/10001.npy', data_1)
    np.save(f'{data_dir}/10002.npy', data_2)

    config = f"""
            load_data:
                inputs: {data_dir}
            build_swc:
              config_kwargs:
                n_states: 3
                learn_means: False
                learn_covariances: True
                window_length: 5
                window_offset: 3
            dual_estimation:
              concatenate: True
                
                """
    labels = [np.array([0, 1]), np.array([1, 2])]
    if not os.path.exists(f'{output_dir}/inf_params/'):
        os.makedirs(f'{output_dir}/inf_params/')
    with open(f'{output_dir}/inf_params/alp.pkl','wb') as file:
        pickle.dump(labels,file)
    with open(f'{output_dir}train_config.yaml', "w") as file:
        yaml.safe_dump(yaml.safe_load(config), file, default_flow_style=False)
    run_pipeline_from_file(f'{output_dir}train_config.yaml', output_dir)

    # After running, check the files in the directory
    means = np.load(f'{output_dir}/dual_estimates/means.npy')
    covs = np.load(f'{output_dir}/dual_estimates/covs.npy')

    covs_answer = np.array([[[2.5, -2.5], [-2.5, 2.5]],
                            [[52., 48.], [48., 52.]],
                            [[54.5, 47.75], [47.75, 54.5]]])

    npt.assert_array_equal(means,np.zeros((3,2)))
    npt.assert_array_equal(covs_answer, covs)

def test_swc_infer_temporal():
    import os
    import pickle
    import shutil
    import yaml
    from osl_dynamics.config_api.pipeline import run_pipeline_from_file

    save_dir = './test_swc_infer_temporal/'
    data_dir = f'{save_dir}/data/'
    output_dir = f'{save_dir}/result/'

    if os.path.isdir(save_dir):
        # Remove the directory
        shutil.rmtree(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_1 = np.array([[1.0, -1.0],
                       [0.0, 0.0],
                       [-1.0, 1.0],
                       [2.0, -2.0],
                       [-2.0, 2.0],
                       [0.0, 0.0],
                       [10.0, 10.0],
                       [-10.0, -10.0],
                       ])
    data_2 = np.array([[2.0, -2.0],
                       [-2.0, 2.0],
                       [0.0, 0.0],
                       [10.0, 10.0],
                       [-10.0, -10.0],
                       [0.0, 3.0],
                       [-3.0, 0.0],
                       [3.0, -3.0]
                       ])
    np.save(f'{data_dir}/10001.npy', data_1)
    np.save(f'{data_dir}/10002.npy', data_2)

    means = np.zeros((3,2))
    covs = np.array([[[2.5, -2.4999995], [-2.4999995, 2.5]],
                     [[52., 48.], [48., 52.]],
                     [[54.5, 47.75], [47.75, 54.5]]])
    inf_params_dir = f'{output_dir}/inf_params/'
    if not os.path.exists(inf_params_dir):
        os.makedirs(inf_params_dir)

    np.save(f'{inf_params_dir}/means.npy',means)
    np.save(f'{inf_params_dir}/covs.npy',covs)

    means_dir = f'{inf_params_dir}/means.npy'
    covs_dir = f'{inf_params_dir}/covs.npy'
    config = f"""
            load_data:
                inputs: {data_dir}
            train_swc:
                config_kwargs:
                    n_channels: 2
                    n_states: 3
                    learn_means: False
                    learn_covariances: False
                    initial_means: {means_dir}
                    initial_covariances: {covs_dir}
                    window_length: 5
                    window_offset: 3
                """

    with open(f'{output_dir}train_config.yaml', "w") as file:
        yaml.safe_dump(yaml.safe_load(config), file, default_flow_style=False)
    run_pipeline_from_file(f'{output_dir}train_config.yaml', output_dir)

    labels_answer = [np.array([0, 1]), np.array([1, 2])]

    with open(f'{inf_params_dir}/alp.pkl','rb') as file:
        labels = pickle.load(file)

    npt.assert_array_equal(labels_answer,labels)

def test_swc_log_likelihood():
    import os
    import json
    import pickle
    import shutil
    import yaml
    from osl_dynamics.config_api.pipeline import run_pipeline_from_file

    save_dir = './test_swc_log_likelihood/'
    data_dir = f'{save_dir}/data/'
    output_dir = f'{save_dir}/result/'

    if os.path.isdir(save_dir):
        # Remove the directory
        shutil.rmtree(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_1 = np.array([[1.0, -1.0],
                       [0.0, 0.0],
                       [-1.0, 1.0],
                       [2.0, -2.0],
                       [-2.0, 2.0],
                       [0.0, 0.0],
                       [10.0, 10.0],
                       [-10.0, -10.0],
                       ])
    data_2 = np.array([[2.0, -2.0],
                       [-2.0, 2.0],
                       [0.0, 0.0],
                       [10.0, 10.0],
                       [-10.0, -10.0],
                       [0.0, 3.0],
                       [-3.0, 0.0],
                       [3.0, -3.0]
                       ])
    np.save(f'{data_dir}/10001.npy', data_1)
    np.save(f'{data_dir}/10002.npy', data_2)

    means = np.zeros((3,2))
    covs = np.array([[[2.5, -2.4999995], [-2.4999995, 2.5]],
                     [[52., 48.], [48., 52.]],
                     [[54.5, 47.75], [47.75, 54.5]]])
    labels = [np.array([0, 1]), np.array([1, 2])]
    inf_params_dir = f'{output_dir}/inf_params/'
    if not os.path.exists(inf_params_dir):
        os.makedirs(inf_params_dir)

    np.save(f'{inf_params_dir}/means.npy',means)
    np.save(f'{inf_params_dir}/covs.npy',covs)

    means_dir = f'{inf_params_dir}/means.npy'
    covs_dir = f'{inf_params_dir}/covs.npy'
    with open(f'{inf_params_dir}/alp.pkl', 'wb') as file:
        pickle.dump(labels, file)

    config = f"""
            load_data:
                inputs: {data_dir}
            build_swc:
                config_kwargs:
                    n_channels: 2
                    n_states: 3
                    learn_means: False
                    learn_covariances: True
                    initial_means: {means_dir}
                    initial_covariances: {covs_dir}
                    window_length: 5
                    window_offset: 3
            log_likelihood:
                static_FC: False
                """

    with open(f'{output_dir}train_config.yaml', "w") as file:
        yaml.safe_dump(yaml.safe_load(config), file, default_flow_style=False)
    run_pipeline_from_file(f'{output_dir}train_config.yaml', output_dir)

    def log_likelihood_calculator(x, mean, cov):
        inv_cov = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)
        n = x.shape[0]
        d = x.shape[1]

        term1, term2 = 0, 0
        for i in range(n):
            diff = x[i] - mean
            term1 += -0.5 * np.log((2 * np.pi) ** d * det_cov)
            term2 += -0.5 * np.dot(diff.T, np.dot(inv_cov, diff))
        return term1 + term2

    # Compute the log-likelihood for each data point and average
    log_likelihood_1 = log_likelihood_calculator(data_1[:5], means[0], covs[0])
    log_likelihood_2 = log_likelihood_calculator(data_1[3:], means[1], covs[1])
    log_likelihood_3 = log_likelihood_calculator(data_2[3:], means[2], covs[2])
    # The answer is total likelihood per session.
    # There are 8 time points per session here. But 20 timepoints calcualted.
    log_likelihood_answer = (log_likelihood_1 + 2 * log_likelihood_2 + log_likelihood_3) / 20 * 8

    with open(f'{output_dir}/metrics.json','r') as file:
        metrics = json.load(file)

    print(log_likelihood_answer)
    npt.assert_almost_equal(log_likelihood_answer,metrics['log_likelihood'])



