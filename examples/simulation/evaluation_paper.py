import os
import numpy as np
from osl_dynamics import simulation
def hmm_iid(save_dir,n_subjects,n_samples,n_states,n_channels):
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

def hmm_hrf():
    pass

def dynemo_iid():
    pass

def dynemo_hrf():
    pass

def swc_iid():
    pass

def swc_hrf():
    pass

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
    }
    if 'hmm_iid' in simulation_list:
        hmm_iid(**config)
    if 'hmm_iid' in simulation_list:
        hmm_hrf()
    if 'dynemo_iid' in simulation_list:
        dynemo_iid()
    if 'dynemo_hrf' in simulation_list:
        dynemo_hrf()
    if 'swc_iid' in simulation_list:
        swc_iid()
    if 'swc_hrf' in simulation_list:
        swc_hrf()
