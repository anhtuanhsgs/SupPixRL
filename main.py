from __future__ import print_function, division
import os, sys, glob, time
import cv2, random
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
import torch.multiprocessing as mp
from environment import *
from utils import get_cell_prob, get_data
from train import train_func
from test import test_func
from natsort import natsorted

from Utils.img_aug_func import *
from Utils.utils import *
from skimage.measure import label
from shared_optim import SharedRMSprop, SharedAdam
from models.models import *

from deploy import deploy
from parser import argparse

def setup_env_conf (args):
    env_conf = {
        "data": args.data,
        "T": args.max_episode_length,
        "size": args.size,
        "fgbg_ratio": args.fgbg_ratio,
        "st_fgbg_ratio": args.st_fgbg_ratio,
        "minsize": args.minsize,
        "no_aug": args.no_aug,

        "3D": "3D" in args.data,
        
        "in_radius": args.in_radius,
        "out_radius": args.out_radius,
        "spl_w": args.spl_w,
        "mer_w": args.mer_w,
        "split": args.split,

        "reward": args.reward,
        "use_lbl": args.use_lbl,
        "use_masks": args.use_masks,
        "DEBUG": args.DEBUG,
        "dilate_fac": args.dilate_fac,

        "tempT": args.max_temp_steps,

        "lowres": args.lowres,
        "T0": args.T0,
        "rew_drop": args.rew_drop,
        "rew_drop_2": args.rew_drop_2,
        

        "exp_pool": args.exp_pool,
    }

    if env_conf ["3D"]:
        env_conf ["size"] = [args.size[2], args.size[0], args.size[1]]

    env_conf ["observation_shape"] = [args.data_channel + 1] + env_conf ["size"]
    


    args.env += "_" + args.model
    env_conf ["data_chan"] = args.data_channel 
    if args.use_lbl:
        # args.env += "_lbl"
        env_conf ["observation_shape"][0] += 1 #Raw, lbl, stop
    if args.use_masks:
        # args.env += "_masks"
        env_conf ["observation_shape"][0] += env_conf ["T"]

    # args.env += "_" + args.data
    
    args.log_dir += args.data + "/" + args.env + "/"
    args.save_model_dir += args.data + "/" + args.env + "/"
    create_dir (args.save_model_dir)
    create_dir (args.log_dir)
    return env_conf

 
def setup_data (args):
    path_test = None
    if args.data == "256_cremi":
        path_train = "Data/Cremi/Corrected/256/train/"
        path_test = "Data/Cremi/Corrected/256/test/"
        path_valid = "Data/Cremi/Corrected/256/valid/"
        args.testlbl = True


    relabel = args.data not in ["256_cremi",]
    
    raw, gt_lbl = get_data (path=path_train, relabel=relabel, data_channel=args.data_channel)
    raw_valid, gt_lbl_valid = get_data (path=path_valid, relabel=relabel, data_channel=args.data_channel)

    raw_test = None
    gt_lbl_test = None
    if path_test is not None:
        raw_test, gt_lbl_test = get_data (path=path_test, relabel=relabel, data_channel=args.data_channel)

    print ("train: ", len (raw), raw [0].shape)
    print ("valid: ", len (raw_valid), raw_valid [0].shape)
    print ("test: ", len (raw_test), raw_test [0].shape)


    raw_test_upsize = None
    gt_lbl_test_upsize = None

    if abs (int (args.downsample) - args.downsample) > 1e-4:
        size = None
        raw = resize_volume (raw, size, ds, "3D" in args.data)
        gt_lbl = resize_volume (gt_lbl, size, ds, "3D" in args.data)
        raw_valid = resize_volume (raw_valid, size, ds, "3D" in args.data)
        gt_lbl_valid = resize_volume (gt_lbl_valid, size, ds, "3D" in args.data)
        args.downsample = 1
    else:
        args.downsample = int (args.downsample)         

    if (args.DEBUG):
        size = [args.size [i] * args.downsample for i in range (len (args.size))]
        if args.downsample == -1:
            size = raw[0].shape[0]
        if "3D" in args.data:
            if args.DEBUG:
                raw = [raw [i] [0:0+size[2], 0:0+size[0], 0:0+size[1]] for i in range (len (raw))]
                gt_lbl = [gt_lbl [i] [0:0+size[2], 0:0+size[0], 0:0+size[1]] for i in range (len (raw))]
                raw = [raw [0]]
                gt_lbl = [gt_lbl [0]]
            else:
                raw = [raw [i] for i in range (len (raw))]
                gt_lbl = [gt_lbl [i] for i in range (len (raw))]
        
        else:    
            raw = [raw [i] [0:0+size,0:0+size] for i in range (20, 21)]
            gt_lbl = [gt_lbl [i] [0:0+size,0:0+size] for i in range (20, 21)]
        raw_valid = np.copy (raw)
        gt_lbl_valid = np.copy (gt_lbl)

    if (args.SEMI_DEBUG):
        raw = raw [:1000]
        gt_lbl = gt_lbl [:1000]

    ds = args.downsample
    if args.downsample:
        size = args.size
        raw = resize_volume (raw, size, ds, "3D" in args.data)
        gt_lbl = resize_volume (gt_lbl, size, ds, "3D" in args.data)
        raw_valid = resize_volume (raw_valid, size, ds, "3D" in args.data)
        gt_lbl_valid = resize_volume (gt_lbl_valid, size, ds, "3D" in args.data)

        if raw_test is not None:
            raw_test = resize_volume (raw_test, size, ds, "3D" in args.data)
        if args.testlbl:
            gt_lbl_test = resize_volume (gt_lbl_test, size, ds, "3D" in args.data)

    return raw, gt_lbl, raw_valid, gt_lbl_valid, raw_test, gt_lbl_test

def main (scripts, args):
    scripts = " ".join (sys.argv[0:])
    args = parser.parse_args()
    args.scripts = scripts
    
    torch.manual_seed(args.seed)
    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method('spawn')

    if (args.deploy):
        raw, gt_lbl, raw_valid, gt_lbl_valid, raw_test, gt_lbl_test = setup_data(args)
    else:
        raw, gt_lbl, raw_valid, gt_lbl_valid, raw_test, gt_lbl_test = setup_data (args)

    env_conf = setup_env_conf (args)


    shared_model = get_model (args, args.model, env_conf ["observation_shape"], args.features, 
                        atrous_rates=args.atr_rate, num_actions=2, split=args.data_channel, 
                        multi=args.multi)

    manager = mp.Manager ()
    shared_dict = manager.dict ()
    if args.wctrl == "s2m":
        shared_dict ["spl_w"] = args.spl_w
        shared_dict ["mer_w"] = args.mer_w

    if args.load:
        saved_state = torch.load(
            args.load,
            map_location=lambda storage, loc: storage)
        shared_model.load_state_dict(saved_state)
    if not args.deploy:
        shared_model.share_memory()

    if args.deploy:
         deploy (shared_model, args, args.gpu_ids [0], (raw_test, gt_lbl_test))
         exit ()
    
    if args.shared_optimizer:
        if args.optimizer == 'RMSprop':
            optimizer = SharedRMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = SharedAdam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
        optimizer.share_memory()
    else:
        optimizer = None


    processes = []
    if not args.no_test:
        if raw_test is not None:
            if (args.deploy):
                p = mp.Process(target=test_func, args=(args, shared_model, env_conf, [raw_valid, gt_lbl_valid], (raw_test, gt_lbl_test, raw_test_upsize, gt_lbl_test_upsize, shared_dict)))
            else:
                p = mp.Process(target=test_func, args=(args, shared_model, env_conf, [raw_valid, gt_lbl_valid], (raw_test, gt_lbl_test), shared_dict))
        else:
            p = mp.Process(target=test_func, args=(args, shared_model, env_conf, [raw_valid, gt_lbl_valid], None, shared_dict))
        p.start()
        processes.append(p)
    
    time.sleep(0.1)

    for rank in range(0, args.workers):
        p = mp.Process(
            target=train_func, args=(rank, args, shared_model, optimizer, env_conf, [raw, gt_lbl], shared_dict))

        p.start()
        processes.append(p)
        time.sleep(0.1)

    for p in processes:
        time.sleep(0.1)
        p.join()

if __name__ == '__main__':
    scripts = " ".join (sys.argv[0:])
    args = parser.parse_args()
    main (scripts, args)
