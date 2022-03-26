from dataloader import make_data_loader
from train import PointNCELossTrainer
import logging
import sys
from parser import get_args

import warnings
warnings.filterwarnings("ignore")

args = get_args()
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s" 
ch = logging.StreamHandler(sys.stdout)
fh = logging.FileHandler(f'{args.exp_name}.log', mode = 'w')
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=[fh, ch])

dataloader = make_data_loader(args)
trainer = PointNCELossTrainer(args, dataloader)
trainer.train()


