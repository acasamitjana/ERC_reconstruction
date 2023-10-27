from os.path import join

from src import losses
from setup_repo import RESULTS_DIR

# Acquisition parameters
HISTO_res = 3.9688e-3  # = 25.4/6400
downsampleFactorHistoLabels = 4
BLOCK_res = 0.1

# Directories
BASE_DIR = join(RESULTS_DIR, 'EX9-19')

# Existing blocks
initial_bid_list_A = ['A1.1', 'A1.2', 'A1.3', 'A1.4',
                      'A2.1', 'A2.2', 'A2.3',
                      'A3.1', 'A3.2', 'A3.3',
                      'A4.1', 'A4.2',
                      'A5.1', 'A5.2',
                      'A6.1'] #15
initial_bid_list_P = ['P1.1', 'P1.2', 'P1.3', 'P1.4',
                      'P2.1', 'P2.2', 'P2.3', 'P2.4',
                      'P3.1', 'P3.2', 'P3.3', 'P3.4',
                      'P4.1', 'P4.2', 'P4.3',
                      'P5.1', 'P5.2', 'P5.3',
                      'P6.1', 'P6.2',
                      'P7.1', 'P7.2',
                      'P8.1',
                      'P9.1'] #24
initial_bid_list_C = ['C1.1', 'C2.1', 'C3.1', 'C4.1', 'C5.1'] #5
initial_bid_list_B = ['B1.1', 'B2.1', 'B3.1', 'B4.1', 'B5.1'] #5

initial_bid_list = initial_bid_list_A + initial_bid_list_P + initial_bid_list_C + initial_bid_list_B
flip_mri_blocks = ['P9.1', 'C5.1']

# Registration params
PROCESSING_SHAPE = (640, 832)

# Linear training
device = 'cuda:0'
training_params = {
    'starting_iteration': 2,
    'n_iterations': 9,
    # Eugenio: new schedule
    'schedule': [0,   2,   0,   3,   0,   4,   0,   4,   5],
    # lbfgs run
    'nepochs': [5, 10, 5, 10, 5, 10, 5, 10, 40],#, 20, 50, 100, 50, 100, 50, 100, 50, 100, 50],
    # LBFGS
    'learning_rate': [0.01,  0.005, 0.01, 0.005, 0.05, 0.005, 0.05, 0.05, 0.2],
    'mask_weight': [5] * 9, # global overlap weight; this multiplies all the structure specific ones
    'mask_cll_weight': [1] * 9,  # cerebellum overlap weight
    'mask_cr_bs_weight': [1] * 9,  # cerebrum overlap weight
    'image_weight': [1] * 9, # intensity loss weight
    'scale_weight': [0.05] * 9,  # scaling loss weight
    # Eugenio: we maybe want to change alpha/beta again in the future if we want to more harshly penalize overlaps (compared with gaps)
    'alpha_weight': [1] * 9, # [2] * 11,  # parameter in overlap weight (for overlaps)
    'beta_weight': [1] * 9,  # parameter in overlap weight (for gaps)
    # Eugenio: select your loss. Global NCC seems to be doing a good job with the intensities
    'loss_intensity': losses.NCC_Loss(device=device, win=[5, 5, 5], name='NCC'),
    # 'loss_intensity': losses.NMI_Loss(device=device, name='NMI', bin_centers=np.linspace(0, 1, 32)),
    # 'loss_intensity': losses.L1_Loss(device=device, name='L1'),
    # 'loss_intensity': losses.GlobalNCC_Loss(device=device, name='GlobalNCC'),
    'cp_spacing': 20  # Eugenio: control point spacing (in pixels) for nonlinear 2D transforms of slices. We should try different values...
}  # training params dict, indicating the different models to be trained, number of epochs and loss weights.

C5_1_vox2ras0 = [
    [-0.001721155596897006, 0.0022033406421542168, -0.7221035957336426, -0.41984533076174557],
    [0.013939060270786285, 0.09974605590105057, 0.013367759063839912, -110.57999075762928],
    [-0.09905922412872314, 0.0070373062044382095, 0.0017798489425331354, -37.58330369065516],
    [0.0, 0.0, 0.0, 1.0]
]

B4_1_vox2ras = [[-0.0033,-0.0999, 0.0045,  41.9693],
                [ -0.0828, 0.0024,-0.1999, -25.3877],
                [0.0560,-0.0023,-0.2751, -77.6671],
                [0.0000, 0.0000, 0.0000,   1.0000]]

B5_1_vox2ras = [[0.0137,-0.0961,-0.0759,  37.8265],
                [0.0824,-0.0026, 0.1768, -55.4398],
                [-0.0550,-0.0277, 0.2462, -80.5761],
                [0.0000, 0.0000, 0.0000,   1.0000]
]