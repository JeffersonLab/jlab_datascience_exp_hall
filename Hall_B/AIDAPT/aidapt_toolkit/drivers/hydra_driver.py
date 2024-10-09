import hydra
import aidapt_toolkit.data_parsers
import aidapt_toolkit.data_prep
import aidapt_toolkit.models
import numpy as np
import matplotlib.pyplot as plt
import os

from omegaconf import OmegaConf
import logging
logging.getLogger('matplotlib').setLevel(logging.INFO)

@hydra.main(version_base=None, config_path="../configs", config_name="hydra_basic_config")
def run(config) -> None:
    save_path = config.driver.save_path
    config = OmegaConf.to_container(config)
    
    vertex_parser = aidapt_toolkit.data_parsers.make('aidapt_numpy_reader_v0', config=config['vertex_parser'], name='vertex_parser')
    detector_parser = aidapt_toolkit.data_parsers.make('aidapt_numpy_reader_v0', config=config['detector_parser'], name='detector_parser')
    lab2inv_prep = aidapt_toolkit.data_prep.make('lab_variables_to_invariants', config=config['lab2inv'])
    d_scaler = aidapt_toolkit.data_prep.make('numpy_minmax_scaler', config=config['d_scaler'], name='detector_scaler')
    v_scaler = aidapt_toolkit.data_prep.make('numpy_minmax_scaler', config=config['v_scaler'], name='vertex_scaler')
    model = aidapt_toolkit.models.make('tf_cgan_v0', config=config['model'], name='mlp_gan_model')

    v_data = vertex_parser.load_data()
    d_data = detector_parser.load_data()

    v_invariants = lab2inv_prep.run(v_data)
    d_invariants = lab2inv_prep.run(d_data)

    d_invariants = d_invariants[:,:-1]

    v_scaler.train(v_invariants)
    d_scaler.train(d_invariants)

    v_invariants_scaled = v_scaler.run(v_invariants)
    d_invariants_scaled = d_scaler.run(d_invariants)

    history = model.train([d_invariants_scaled, v_invariants_scaled])
    d_results_scaled = model.predict(v_invariants_scaled)

    d_results = d_scaler.reverse(d_results_scaled)

    # Plot training history
    model.analysis(save_path)

    # Plot distributions
    fig, axs = plt.subplots(2,2)
    axs = axs.flat
    output_names = ('sppim', 'spipm', 'tpip', 'alpha')
    for i, (ax, name) in enumerate(zip(axs, output_names)):
        ax.hist(d_results[:,i], bins=100,histtype='step',color="blue",density=True, label='GAN')
        ax.hist(d_invariants[:,i], bins=100,histtype='step',color="orange",density=True, label='Detector')
        ax.set_xlabel(name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, 'distributions.png'))

    # Plot reconstruction errors
    fig, ax = plt.subplots(1,1)
    p_rec_gan = np.sqrt((np.power(d_results[:,0],2) + np.power(d_results[:,1],2) + np.power(d_results[:,2],2) ) )
    p_gen = np.sqrt((np.power(v_invariants[:,0],2) + np.power(v_invariants[:,1],2) + np.power(v_invariants[:,2],2) ) )
    p_rec = np.sqrt((np.power(d_invariants[:,0],2) + np.power(d_invariants[:,1],2) + np.power(d_invariants[:,2],2) ) )
    ax.hist(p_rec - p_gen, bins = 100, histtype = 'step')
    ax.hist(p_rec_gan - p_gen, bins = 100, histtype = 'step')
    ax.legend([r'$p_{rec} - p_{gen}$', r'$p_{rec~GAN} - p_{gen}$'], fontsize = 15, frameon = 0, loc= "upper left")
    fig.savefig(os.path.join(save_path, 'reconstruction_errors.png'))

    vertex_parser.save(os.path.join(save_path, 'vertex_parser'))
    detector_parser.save(os.path.join(save_path, 'detector_parser'))
    lab2inv_prep.save(os.path.join(save_path, 'lab2inv_prep'))
    d_scaler.save(os.path.join(save_path, 'd_scaler'))
    v_scaler.save(os.path.join(save_path, 'v_scaler'))
    model.save(os.path.join(save_path, 'cgan_model'))

if __name__ == "__main__":
    run()
