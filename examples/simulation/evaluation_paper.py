import os
import numpy as np
from osl_dynamics import simulation
from osl_dynamics.array_ops import apply_hrf
def hmm_iid(save_dir,n_subjects,n_samples,n_states,n_channels,tr):
    save_dir = f'{save_dir}/hmm_iid/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth/')

    sim = simulation.HMM_MVN(
        n_samples=n_samples * n_subjects,
        n_states=n_states,
        n_channels=n_channels,
        trans_prob='uniform',
        stay_prob=0.9,
        means='zero',
        covariances='random'
    )
    data = sim.time_series
    time_course = sim.state_time_course
    data = data.reshape(n_subjects, -1, n_channels)
    time_course = time_course.reshape(n_subjects, -1, n_states)

    np.save(f'{save_dir}truth/state_covariances.npy', sim.obs_mod.covariances)
    np.save(f'{save_dir}truth/tpm.npy', sim.hmm.trans_prob)

    for i in range(n_subjects):
        np.savetxt(f'{save_dir}{10001 + i}.txt', data[i])
        np.save(f'{save_dir}truth/{10001 + i}_state_time_course.npy', time_course[i])

def hmm_hrf(save_dir,n_subjects,n_samples,n_states,n_channels,tr):
    save_dir = f'{save_dir}/hmm_hrf/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth/')

    sim = simulation.HMM_MVN(
        n_samples=n_samples * n_subjects,
        n_states=n_states,
        n_channels=n_channels,
        trans_prob='uniform',
        stay_prob=0.9,
        means='zero',
        covariances='random'
    )
    data = sim.time_series
    time_course = sim.state_time_course
    data = data.reshape(n_subjects, -1, n_channels)
    time_course = time_course.reshape(n_subjects, -1, n_states)

    np.save(f'{save_dir}truth/state_covariances.npy', sim.obs_mod.covariances)
    np.save(f'{save_dir}truth/tpm.npy', sim.hmm.trans_prob)

    for i in range(n_subjects):
        np.savetxt(f'{save_dir}{10001 + i}.txt', apply_hrf(data[i],tr))
        np.save(f'{save_dir}truth/{10001 + i}_state_time_course.npy', time_course[i])


def dynemo_iid(save_dir,n_subjects,n_samples,n_states,n_channels,tr):
    save_dir = f'{save_dir}/dynemo_iid/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth/')
    sim = simulation.MixedSine_MVN(
        n_samples=n_subjects * n_samples,
        n_modes=n_states,
        n_channels=n_channels,
        relative_activation=[1, 0.5, 0.5, 0.25, 0.25, 0.2],
        amplitudes=[6, 5, 4, 3, 2, 1],
        frequencies=[1, 2, 3, 4, 6, 8],
        sampling_frequency=250,
        means="zero",
        covariances="random",
    )

    data = sim.time_series
    time_course = sim.mode_time_course
    data = data.reshape(n_subjects, -1, n_channels)
    time_course = time_course.reshape(n_subjects, -1, n_states)

    np.save(f'{save_dir}truth/state_covariances.npy', sim.obs_mod.covariances)

    for i in range(n_subjects):
        np.savetxt(f'{save_dir}{10001 + i}.txt', data[i])
        np.save(f'{save_dir}truth/{10001 + i}_mode_time_course.npy', time_course[i])

def dynemo_hrf(save_dir,n_subjects,n_samples,n_states,n_channels,tr):
    save_dir = f'{save_dir}/dynemo_hrf/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth/')
    sim = simulation.MixedSine_MVN(
        n_samples=n_subjects * n_samples,
        n_modes=n_states,
        n_channels=n_channels,
        relative_activation=[1, 0.5, 0.5, 0.25, 0.25, 0.2],
        amplitudes=[6, 5, 4, 3, 2, 1],
        frequencies=[1, 2, 3, 4, 6, 8],
        sampling_frequency=250,
        means="zero",
        covariances="random",
    )

    data = sim.time_series
    time_course = sim.mode_time_course
    data = data.reshape(n_subjects, -1, n_channels)
    time_course = time_course.reshape(n_subjects, -1, n_states)

    np.save(f'{save_dir}truth/state_covariances.npy', sim.obs_mod.covariances)

    for i in range(n_subjects):
        np.savetxt(f'{save_dir}{10001 + i}.txt', apply_hrf(data[i],tr))
        np.save(f'{save_dir}truth/{10001 + i}_mode_time_course.npy', time_course[i])

def swc_iid(save_dir,n_subjects,n_samples,n_states,n_channels,tr):
    save_dir = f'{save_dir}/swc_iid/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth/')
    sim = simulation.SWC_MVN(
        n_samples=n_samples * n_subjects,
        n_states=n_states,
        n_channels=n_channels,
        stay_time=100,
        means="zero",
        covariances='random'
    )
    data = sim.time_series
    time_course = sim.state_time_course
    data = data.reshape(n_subjects, -1, n_channels)
    time_course = time_course.reshape(n_subjects, -1, n_states)

    np.save(f'{save_dir}truth/state_covariances.npy', sim.covariances)

    for i in range(n_subjects):
        np.savetxt(f'{save_dir}{10001 + i}.txt', data[i])
        np.save(f'{save_dir}truth/{10001 + i}_state_time_course.npy', time_course[i])

def swc_hrf(save_dir,n_subjects,n_samples,n_states,n_channels,tr):
    save_dir = f'{save_dir}/swc_hrf/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(f'{save_dir}truth/'):
        os.makedirs(f'{save_dir}truth/')
    sim = simulation.SWC_MVN(
        n_samples=n_samples * n_subjects,
        n_states=n_states,
        n_channels=n_channels,
        stay_time=100,
        means="zero",
        covariances='random'
    )
    data = sim.time_series
    time_course = sim.state_time_course
    data = data.reshape(n_subjects, -1, n_channels)
    time_course = time_course.reshape(n_subjects, -1, n_states)

    np.save(f'{save_dir}truth/state_covariances.npy', sim.covariances)

    for i in range(n_subjects):
        np.savetxt(f'{save_dir}{10001 + i}.txt', apply_hrf(data[i],tr))
        np.save(f'{save_dir}truth/{10001 + i}_state_time_course.npy', time_course[i])

def main(simulation_list=None):
    save_dir = './data/node_timeseries/simulation_final/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    config = {
        'save_dir': save_dir,
        'n_subjects':500,
        'n_states': 6,
        'n_channels': 50,
        'n_samples':1200,
        'tr':0.72
    }

    if 'hmm_iid' in simulation_list:
        hmm_iid(**config)
    if 'hmm_iid' in simulation_list:
        hmm_hrf(**config)
    if 'dynemo_iid' in simulation_list:
        dynemo_iid(**config)
    if 'dynemo_hrf' in simulation_list:
        dynemo_hrf(**config)
    if 'swc_iid' in simulation_list:
        swc_iid(**config)
    if 'swc_hrf' in simulation_list:
        swc_hrf(**config)
