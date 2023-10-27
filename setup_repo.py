from os.path import join, exists
from os import makedirs

BASE_DIR = '/media/biofisica/HDD_ADRIA'#'/home/acasamitjana'
REPO_DIR = join(BASE_DIR, 'Repositories', 'ERC_reconstruction')
DATA_DIR = join(BASE_DIR, 'Data', 'BUNGEE_TOOLS')
RESULTS_DIR = join(BASE_DIR, 'Results', 'BUNGEE_Tools')
WEBPAGE_DIR = join(BASE_DIR, 'ERC-project-webviz')
if not exists(RESULTS_DIR):
    makedirs(RESULTS_DIR)

NIFTY_REG_DIR = '/home/acasamitjana/Software_MI/niftyreg-git/build/'
ALADINcmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_aladin'
F3Dcmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_f3d'
TRANSFORMcmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_transform'
REScmd = NIFTY_REG_DIR + 'reg-apps' + '/reg_resample'
