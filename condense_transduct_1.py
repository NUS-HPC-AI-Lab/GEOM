from deeprobust.graph.data import Dataset
import numpy as np
import random
import time
import argparse
import torch
from utils import *
import torch.nn.functional as F
from geomtt_transduct1 import MetaGtt
from utils_graphsaint import DataGraphSAINT
import logging
import sys
import datetime
import os
from tensorboardX import SummaryWriter
import deeprobust.graph.utils as utils
import json


parser = argparse.ArgumentParser(description="Distill-Step:distillation the graph dataset into structure-free node set.")
parser.add_argument("--config", type=str, default="", help="Path to the config JSON file")
parser.add_argument("--section", type=str, default='runed exps name', help="the experiments needs to run")
parser.add_argument('--device', type=str, default='cuda:5')
parser.add_argument('--student_nlayers', type=int, default=2)
parser.add_argument('--student_hidden', type=int, default=256)
parser.add_argument('--student_dropout', type=float, default=0.0)
parser.add_argument('--seed_student', type=int, default=15, help='Random seed for distill student model')
parser.add_argument('--save_log', type=str, default='logs', help='path to save logs')
parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")  #默认是false
parser.add_argument('--max_files', type=int, default=None,
                    help='number of expert files to read (leave as None unless doing ablations)')
parser.add_argument('--max_experts', type=int, default=None,
                    help='number of experts to read per file (leave as None unless doing ablations)')
parser.add_argument('--condense_model', type=str, default='GCN', help='Default condensation model')
parser.add_argument('--eval_model', type=str, default='SGC', help='evaluation model for saving best feat')
parser.add_argument('--eval_type', type=str, default='S',help='eval_mode, check utils.py for more info')
parser.add_argument('--initial_save', type=int, default=0, help='whether save initial feat and syn')
parser.add_argument('--interval_buffer', type=int, default=1, choices=[0,1],help='whether use interval buffer')
parser.add_argument('--rand_start', type=int, default=1,choices=[0,1], help='whether use random start')
parser.add_argument('--optimizer_con', type=str, default='Adam', help='See choices', choices=['Adam', 'SGD'])
parser.add_argument('--optim_lr', type=int, default=0, help='whether use LR lr learning optimizer')
parser.add_argument('--optimizer_lr', type=str, default='Adam', help='See choices', choices=['Adam', 'SGD'])
parser.add_argument('--coreset_method', type=str, default='kcenter')
parser.add_argument('--coreset_seed', type=int, default=15)
parser.add_argument('--max_start_epoch_s', type=int, default=1) 
parser.add_argument('--max_start_epoch', type=int, default=5)    
parser.add_argument('--min_start_epoch', type=int, default=0)     
parser.add_argument('--nruns', type=int, default=1)
parser.add_argument('--whole_data', type=int, default=0)
parser.add_argument('--test_model_iters', type=int, default=200)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--test_model_type', type=str, default='GCN')



parser.add_argument('--lam', default=0.75)
parser.add_argument('--T', default=1500)
parser.add_argument('--scheduler', default='root')
parser.add_argument('--gpuid', default=0)

parser.add_argument('--expert_epochs', type=int, default=100)
parser.add_argument('--syn_steps', type=int, default=100)
parser.add_argument('--lr_feat', type=float, default=0.01)

parser.add_argument('--lr_y', type=float, default=0.001)
parser.add_argument('--beta', type=float, default=0.5)

parser.add_argument("--soft_label", action='store_true')

args = parser.parse_args()


with open(args.config, "r") as config_file:
    config = json.load(config_file)

if args.section in config:
    section_config = config[args.section]

for key, value in section_config.items():
    setattr(args, key, value)

log_dir = './' + args.save_log + '/Distill/norm-2-val-kd2-lr_y{}-{}-reduce_{}-lam-{}-T-{}-scheduler-{}-min{}-max{}-syn_steps{}-expert_epochs{}-lr_feat{}-max_start_epoch_s{}'.format(args.lr_y,args.dataset, str(args.reduction_rate),args.lam,args.T,args.scheduler
                                                                                            ,args.min_start_epoch,args.max_start_epoch,args.syn_steps,args.expert_epochs,args.lr_feat,args.max_start_epoch_s)

args.device = f"cuda:{args.gpuid}"


device = torch.device(args.device)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info('This is the log_dir: {}'.format(log_dir))


# random seed setting
random.seed(args.seed_student)
np.random.seed(args.seed_student)
torch.manual_seed(args.seed_student)
torch.cuda.manual_seed(args.seed_student)
device = torch.device(args.device)
logging.info('args = {}'.format(args))

data_graphsaint = ['flickr', 'reddit', 'ogbn-arxiv']
if args.dataset in data_graphsaint:
    data = DataGraphSAINT(args.dataset)
    data_full = data.data_full
else:
    data_full = get_dataset(args.dataset)
    data = Transd2Ind(data_full)
args.log_dir = log_dir
args.buffer_path= './logs/Buffer/{}-lam-{}-T-{}-scheduler-{}'.format(args.dataset,args.lam,args.T,args.scheduler)


agent = MetaGtt(data, args, device=device)
writer = SummaryWriter(log_dir + '/tbx_log')

if args.dataset == 'ogbn-arxiv':
    args.soft_label= True
    

agent.distill(writer)


logging.info('Finish! Log_dir: {}'.format(log_dir))
