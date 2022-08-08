import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--logging_dir")
parser.add_argument("--org")
args = parser.parse_args()

print(args.logging_dir)


hidden_layer_size = 512

aggregate_path_length = 4

merge_peak_tolerance = 0.1

max_seq_len = 32

max_num_peaks = 400

learning_rate = 0.0001

max_global_norm = 5

val_stack_size = 100
train_stack_size = 80
test_stack_size = 100

train_batch_size = 4
test_batch_size = 8

l2_lambda = 0.000001

num_features = 4

num_ion = 8

num_ion_tnet = 12

AA_mass_tolerance = 0.05

########################################################################
# VOCABULARY 
########################################################################


_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_START_VOCAB = [_PAD, _GO, _EOS]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2

vocab_reverse = ['A',
         'R',
         'N',
        'Nmod',
         'D',
         # 'C',
        'Cmod',
         'E',
         'Q',
        'Qmod',
         'G',
         'H',
        'I',
         'L',
         'K',
         'M',
        'Mmod',
         'F',
         'P',
         'S',
         'T',
         'W',
         'Y',
         'V',
        ]

vocab_reverse = _START_VOCAB + vocab_reverse

vocab = dict([(x, y) for (y, x) in enumerate(vocab_reverse)])

vocab_size = len(vocab_reverse)

########################################################################
# MASS
########################################################################

mass_H = 1.0078
mass_H2O = 18.0106
mass_NH3 = 17.0265
mass_N_terminus = 1.0078
mass_C_terminus = 17.0027
mass_CO = 27.9949

mass_AA = {'_PAD':0.0,
         '_GO':mass_N_terminus-mass_H,
         '_EOS':mass_C_terminus+mass_H,
         'A':71.03711, # 0
         'R':156.10111, # 1
         'N':114.04293, # 2
         'Nmod':115.02695,
         'D':115.02694, # 3
         # 'C': 103.00919, # 4
         'Cmod':160.03065, # C(+57.02)
         #~ 'Cmod':161.01919, # C(+58.01) # orbi
         'E':129.04259, # 5
         'Q':128.05858, # 6
         'Qmod':129.0426,
         'G':57.02146, # 7
         'H':137.05891, # 8
         'I':113.08406, # 9
         'L':113.08406, # 10
         'K':128.09496, # 11
         'M':131.04049, # 12
         'Mmod':147.0354,
         'F':147.06841, # 13
         'P':97.05276, # 14
         'S':87.03203, # 15
         'T':101.04768, # 16
         'W':186.07931, # 17
         'Y':163.06333, # 18
         'V':99.06841, # 19
        }

mass_ID = [mass_AA[vocab_reverse[x]] for x in range(vocab_size)]
mass_ID_np = np.array(mass_ID, dtype=np.float32)

mass_AA_min = mass_AA["G"] # 57.02146

WINDOW_SIZE = 10


MZ_MAX = 3000.0

PRECURSOR_MASS_PRECISION_INPUT_FILTER = 1000



if args.org=="yeast":
    train_file = "data//cross.9high_80k.exclude_yeast/cross.cat.mgf.train.repeat"
    val_file = "data/cross.9high_80k.exclude_yeast/cross.cat.mgf.valid.repeat"
    
    train_spec_loc =np.load("data/cross.9high_80k.exclude_yeast/cross.cat.mgf.train.repeat_spec_loc.npy")
    val_spec_loc = np.load("data/cross.9high_80k.exclude_yeast/cross.cat.mgf.valid.repeat_spec_loc.npy")

    test_file = "data/SpeciesTestSets/Yeast_peaks.db.10k.mgf"
    

# ############## Human
if args.org=="human":
    train_file = "data/cross.9high_80k.exclude_human/cross.cat.mgf.train.repeat"
    val_file = "data/cross.9high_80k.exclude_human/cross.cat.mgf.valid.repeat"
    
    train_spec_loc =np.load("data/cross.9high_80k.exclude_human/cross.cat.mgf.train.repeat_spec_loc.npy")
    val_spec_loc = np.load("data/cross.9high_80k.exclude_human/cross.cat.mgf.valid.repeat_spec_loc.npy")

    test_file = "data/SpeciesTestSets/Human_peaks.db.10k.mgf"


# ############## mouse
if args.org=="mouse":
    train_file = "data/cross.9high_80k.exclude_mouse/cross.cat.mgf.train.repeat"
    val_file = "data/cross.9high_80k.exclude_mouse/cross.cat.mgf.valid.repeat"
    
    train_spec_loc =np.load("data/cross.9high_80k.exclude_mouse/cross.cat.mgf.train.repeat_spec_loc.npy")
    val_spec_loc = np.load("data/cross.9high_80k.exclude_mouse/cross.cat.mgf.valid.repeat_spec_loc.npy")

    test_file = "data/SpeciesTestSets/Mouse_peaks.db.10k.mgf"


# ############## bacillus
if args.org=="bacillus":
    train_file = "data/cross.9high_80k.exclude_bacillus/cross.cat.mgf.train.repeat"
    val_file = "data/cross.9high_80k.exclude_bacillus/cross.cat.mgf.valid.repeat"
    
    train_spec_loc =np.load("data/cross.9high_80k.exclude_bacillus/cross.cat.mgf.train.repeat_spec_loc.npy")
    val_spec_loc = np.load("data/cross.9high_80k.exclude_bacillus/cross.cat.mgf.valid.repeat_spec_loc.npy")

    test_file = "data/SpeciesTestSets/Bacillus_peaks.db.10k.mgf"


############## clambacteria
if args.org=="clam":    
    train_file = "data/cross.9high_80k.exclude_clambacteria/cross.cat.mgf.train.repeat"
    val_file = "data/cross.9high_80k.exclude_clambacteria/cross.cat.mgf.valid.repeat"
    
    train_spec_loc =np.load("data/cross.9high_80k.exclude_clambacteria/cross.cat.mgf.train.repeat_spec_loc.npy")
    val_spec_loc = np.load("data/cross.9high_80k.exclude_clambacteria/cross.cat.mgf.valid.repeat_spec_loc.npy")

    test_file = "data/SpeciesTestSets/Clambacteria_peaks.db.10k.mgf"


# ############## honeybee
if args.org=="honeybee":
    train_file = "data/cross.9high_80k.exclude_honeybee/cross.cat.mgf.train.repeat"
    val_file = "data/cross.9high_80k.exclude_honeybee/cross.cat.mgf.valid.repeat"
    
    train_spec_loc =np.load("data/cross.9high_80k.exclude_honeybee/cross.cat.mgf.train.repeat_spec_loc.npy")
    val_spec_loc = np.load("data/cross.9high_80k.exclude_honeybee/cross.cat.mgf.valid.repeat_spec_loc.npy")

    test_file = "data/SpeciesTestSets/Honeybee_peaks.db.10k.mgf"



# ############## ricebean
if args.org=="ricebean":
    train_file = "data/cross.9high_80k.exclude_ricebean/cross.cat.mgf.train.repeat"
    val_file = "data/cross.9high_80k.exclude_ricebean/cross.cat.mgf.valid.repeat"
    
    train_spec_loc =np.load("data/cross.9high_80k.exclude_ricebean/cross.cat.mgf.train.repeat_spec_loc.npy")
    val_spec_loc = np.load("data/cross.9high_80k.exclude_ricebean/cross.cat.mgf.valid.repeat_spec_loc.npy")

    test_file = "data/SpeciesTestSets/Ricebean_peaks.db.10k.mgf"



# ############## tomato
if args.org=="tomato":
    train_file = "data/cross.9high_80k.exclude_tomato/cross.cat.mgf.train.repeat"
    val_file = "data/cross.9high_80k.exclude_tomato/cross.cat.mgf.valid.repeat"
    
    train_spec_loc =np.load("data/cross.9high_80k.exclude_tomato/cross.cat.mgf.train.repeat_spec_loc.npy")
    val_spec_loc = np.load("data/cross.9high_80k.exclude_tomato/cross.cat.mgf.valid.repeat_spec_loc.npy")

    test_file = "data/SpeciesTestSets/Tomato_peaks.db.10k.mgf"



# ############## mmazei
if args.org=="mmazei":
    train_file = "data/cross.9high_80k.exclude_mmazei/cross.cat.mgf.train.repeat"
    val_file = "data/cross.9high_80k.exclude_mmazei/cross.cat.mgf.valid.repeat"
    
    train_spec_loc =np.load("data/cross.9high_80k.exclude_mmazei/cross.cat.mgf.train.repeat_spec_loc.npy")
    val_spec_loc = np.load("data/cross.9high_80k.exclude_mmazei/cross.cat.mgf.valid.repeat_spec_loc.npy")

    test_file = "data/SpeciesTestSets/Mmazei_peaks.db.10k.mgf"


if args.org=="yeastNR0":
    test_file = "/media/kevin/One Touch/PhD/SpecData/Prosit/Yeast/yeastRandom10k.mgf"
if args.org=="yeastNR1":
    test_file = "/media/kevin/One Touch/PhD/SpecData/Prosit/Yeast/yeastRandom10kNR1.mgf"
if args.org=="yeastNR5":
    test_file = "/media/kevin/One Touch/PhD/SpecData/Prosit/Yeast/yeastRandom10kNR5.mgf"
if args.org=="yeastNR10":
    test_file = "/media/kevin/One Touch/PhD/SpecData/Prosit/Yeast/yeastRandom10kNR10.mgf"
if args.org=="yeastNR15":
    test_file = "/media/kevin/One Touch/PhD/SpecData/Prosit/Yeast/yeastRandom10kNR15.mgf"
    
    
data_type = "mgf"
