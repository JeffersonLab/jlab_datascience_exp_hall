import os
from Hall_B.AIDAPT.utils.graph_driver_utils import GraphRuntime
print(os.getcwd())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Name, registered ID pairs
modules = {
    'vertex_parser': 'aidapt_numpy_reader_v0',
    'detector_parser': 'aidapt_numpy_reader_v0',
    'lab2inv': 'lab_variables_to_invariants',
    'v_scaler': 'numpy_minmax_scaler',
    'd_scaler': 'numpy_minmax_scaler',
    'model': 'tf_mlp_gan_v0'
}

### Graph Example ###
# Ordered (input, module.function, output) tuples
# Has weird syntax if your input is a tuple that goes to one argument 
# (e.g. model.train takes x and y as a single tuple data=(x,y))
graph = [
    (None, 'vertex_parser.load_data', 'v_data'),
    (None, 'detector_parser.load_data', 'd_data'),
    ('v_data', 'lab2inv.run', 'v_invariants'),
    ('d_data', 'lab2inv.run', 'd_invariants'),
    ('v_invariants', 'v_scaler.train', None),
    ('d_invariants', 'd_scaler.train', None),
    ('v_invariants', 'v_scaler.run', 'v_scaled'),
    ('d_invariants', 'd_scaler.run', 'd_scaled'),
    (('d_scaled', 'v_scaled'), 'combine', 'dv_list'),
    ('dv_list', 'model.train', 'metrics'),
    (None, 'model.analysis', None),
    ('v_scaled', 'model.predict', 'd_preds_scaled'),
    ('d_preds_scaled', 'd_scaler.reverse', 'd_preds'),
]

### Config containing configuration dictionaries for all modules in `modules` ###
### Should contain a key for each module based on it's name (not registered ID)
config = {
    'vertex_parser': {
        'filepaths': [
            './scratch/data/ps_vertex/ps_vertex_0.npy',
            './scratch/data/ps_vertex/ps_vertex_1.npy',
            './scratch/data/ps_vertex/ps_vertex_2.npy',
            './scratch/data/ps_vertex/ps_vertex_3.npy']},
    'detector_parser': {
        'filepaths': [
            './scratch/data/ps_detector/ps_detector_0.npy',
            './scratch/data/ps_detector/ps_detector_1.npy',
            './scratch/data/ps_detector/ps_detector_2.npy',
            './scratch/data/ps_detector/ps_detector_3.npy']},
    'lab2inv': {},
    'v_scaler': {'feature_range': (-1,1)},
    'd_scaler': {'feature_range': (-1,1)},
    'model': {
        'generator_optimizer': ('Adam', {'learing_rate': 5e-5, 'beta_1': 0.5}),
        'discriminator_optimizer': ('Adam', {'learing_rate': 1e-5, 'beta_1': 0.5}),
        'latent_dim': 100,
        'label_shape': 5,
        'image_shape': 5,
        'generator_layers': [
            ('Dense', {'units': 64}),
            ('LeakyReLU', {'alpha': 0.2}),
            ('BatchNormalization', {'momentum': 0.8}),
            ('Dense', {'units': 128}),
            ('LeakyReLU', {'alpha': 0.2}),
            ('BatchNormalization', {'momentum': 0.8}),
            ('Dense', {'units': 256}),
            ('LeakyReLU', {'alpha': 0.2}),
            ('BatchNormalization', {'momentum': 0.8})],
        'discriminator_layers': [
            ('Dense', {'units': 256}),
            ('LeakyReLU', {'alpha': 0.2}),
            ('Dense', {'units': 128}),
            ('LeakyReLU', {'alpha': 0.2}),
            ('Dense', {'units': 64}),
            ('LeakyReLU', {'alpha': 0.2})],
        'epochs': 10,
        'batch_size': 512,
    },
}



gr = GraphRuntime()

gr.run_graph(graph, modules, config)