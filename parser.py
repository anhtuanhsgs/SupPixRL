import argparse

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument(
    '--env',
    default='EM_env',
    metavar='ENV',
    )

parser.add_argument(
    '--lr',
    type=float,
    default=0.0001,
    metavar='LR',
    help='learning rate (default: 0.0001)')

parser.add_argument(
    '--gamma',
    type=float,
    default=1,
    metavar='G',
    help='discount factor for rewards (default: 1)')

parser.add_argument(
    '--tau',
    type=float,
    default=1.00,
    metavar='T',
    help='parameter for GAE (default: 1.00)')

parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')

parser.add_argument(
    '--workers',
    type=int,
    default=8,
    metavar='W',
    help='how many training workers to use (default: 8)')

parser.add_argument(
    '--num-steps',
    type=int,
    default=5,
    metavar='NS',
    help='n in n-step learning A3C')

parser.add_argument(
    '--max-episode-length',
    type=int,
    default=5,
    metavar='M',
    help='Maximum number of coloring steps')

parser.add_argument(
    '--save-max',
    default=True,
    metavar='SM',
    help='Save model on every test run high score matched or bested')

parser.add_argument(
    '--load', default=False, metavar='L', help='load a trained model')

parser.add_argument(
    '--optimizer',
    default='Adam',
    metavar='OPT',)

parser.add_argument(
    '--load-model-dir',
    default='../trained_models/',
    metavar='LMD',
    help='folder to load trained models from')

parser.add_argument(
    '--save-model-dir',
    default='logs/trained_models/',
    metavar='SMD',
    help='folder to save trained models')

parser.add_argument(
    '--log-dir', default='logs/', metavar='LG', help='folder to save logs')

parser.add_argument(
    '--gpu-ids',
    type=int,
    default=-1,
    nargs='+',
    help='GPUs to use [-1 CPU only] (default: -1)')

parser.add_argument(
    '--amsgrad',
    default=True,
    metavar='AM',
    help='Adam optimizer amsgrad parameter')

parser.add_argument(
    '--save-period',
    type=int,
    default=100,
    metavar='SP',
    help='Save period')

parser.add_argument(
    '--log-period',
    type=int,
    default=10,
    metavar='LP',
    help='Log period')

parser.add_argument (
    '--train-log-period',
    type=int,
    default=16,
    metavar='TLP',
)

parser.add_argument(
    '--shared-optimizer',
    action='store_true'
)


parser.add_argument (
    '--recur-feat',
    type=int,
    default=64,
    metavar='HF'
)

parser.add_argument (
    '--in-radius',
    type=float,
    default=[1.1],
    nargs='+',
    help='Splitting radii r'
)

parser.add_argument (
    '--out-radius',
    type=int,
    default=[16, 48, 96],
    nargs='+',
    help='Shrinking factors alpha'
)

parser.add_argument (
    '--eval-data',
    default='test',
    choices=['test', 'valid', 'train', 'all']
)

parser.add_argument (
    '--merge_radius',
    type=int,
    default=[16, 48, 96],
    nargs='+'
)

parser.add_argument (
    '--split_radius',
    type=int,
    default=[16, 48, 96],
    nargs='+', 
)

parser.add_argument (
    '--merge_speed',
    type=int,
    default=[1, 2, 4],
    nargs='+'
)

parser.add_argument (
    '--split_speed',
    type=int,
    default=[1, 2, 4],
    nargs='+'
)

parser.add_argument (
    '--features',
    type=int,
    default= [32, 64, 128, 256],
    nargs='+', 
    help='Feature size of the core network'
)

parser.add_argument (
    '--size',
    type=int,
    default= [96, 96],
    nargs='+',
    help='Input processing size of agent, if processing size is smaller than size of images in the data, input will be cropped from the image',
)

parser.add_argument (
    '--entropy-alpha',
    type=float,
    default=0.5,
)

parser.add_argument (
    '--model',
    default='UNet',
    choices=["AttUNet", "ASPPAttUNet", "DeepLab", "ASPPAttUNet2", 
                        "AttUNet2", "AttUNet3", "GCNAttUNet", "UNet3D", "UNet2D"]
)

parser.add_argument (
    "--reward",
    default="normal",
    choices=["normal", "gaussian", "density", "seg"]
)

parser.add_argument (
    "--use-lbl",
    action="store_true"
)

parser.add_argument (
    "--use-masks",
    action="store_true"
)

parser.add_argument (
    '--downsample',
    type=float,
    default=1,
    help='Data down-sampling rate, -1 to have all images down-sampled to input processing size of agent',
)

parser.add_argument (
    '--DEBUG',
    action="store_true"
)

parser.add_argument (
    '--data',
    default='snemi',
    choices=['syn', 'snemi', 'voronoi', 'zebrafish', 'cvppp', 'cvppp_eval', 'zebrafish3D', 'dic-hela', 
            'sb2018', 'kitti', '160_mnseg2018', '224_mnseg2018', "256_cremi", "448_cremi", '96_cremi3D', '64_cremi3D', "ctDNA", "cremi3D", "160_cremi"]
)

parser.add_argument (
    '--SEMI_DEBUG',
    action="store_true"
)

parser.add_argument (
    '--deploy',
    action='store_true',
    help='Enable for test set deployment',
)

parser.add_argument (
    '--fgbg-ratio',
    default=0.2,
    type=float,
)

parser.add_argument (
    '--st-fgbg-ratio',
    default=0.5,
    type=float,
)

parser.add_argument (
    '--seg-scale',
    action='store_true'
)

# parser.add_argument (
#     '--lbl-action-ratio',
#     type=float,
#     default=0.125
# )

parser.add_argument (
    '--lbl-agents',
    type=int,
    default=0
)

parser.add_argument (
    '--minsize',
    type=int,
    default=0,
)


parser.add_argument (
    '--spl_w',
    type=float,
    default=2,
    help='Splitting weight w_s',
)

parser.add_argument (
    '--mer_w',
    type=float,
    default=1,
    help='Merging weight w_s',
)

parser.add_argument (
    '--noisy',
    action='store_true',
)

parser.add_argument (
    '--lstm-feats',
    type=int,
    default=0,
)

parser.add_argument (
    '--valid-gpu',
    type=int,
    default=-1,
    help='Choose gpu-id for the verbose worker',
)

parser.add_argument (
    '--atr-rate',
    type=int,
    default= [1, 6, 12, 18],
    nargs='+',
    help='Attrous spatial pooling rates',
)

parser.add_argument (
    '--dilate-fac',
    type=int,
    default=2
)

parser.add_argument (
    '--lowres',
    action="store_true"
)

parser.add_argument (
    '--no-aug',
    action="store_true"
)

parser.add_argument (
    '--multi',
    type=int,
    default=1,
)

parser.add_argument (
    '--max-temp-steps',
    type=int,
    default=99,
)

parser.add_argument (
    '--T0',
    type=int,
    default=0
)

parser.add_argument (
    "--no-test",
    action="store_true"
)

parser.add_argument (
    '--rew-drop',
    type=int,
    default=0
)

parser.add_argument (
    '--rew-drop-2',
    type=int,
    default=0
)


parser.add_argument (
    '--split',
    default='ins',
    choices=['ins', 'prox',]
)

parser.add_argument (
    '--wctrl',
    default='non',
    choices=['non', 's2m']
)

parser.add_argument (
    '--wctrl-schedule',
    default=[2000, 4000, 6000, 8000, 10000, 12000, 14000],
    type=int,
    nargs='+',
)

parser.add_argument (
    '--exp-pool',
    default=0,
    type=int,
)