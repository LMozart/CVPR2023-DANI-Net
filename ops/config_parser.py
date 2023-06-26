import os
import yaml
import argparse
from utils.utils import *
from cfgnode import CfgNode

class ExpConfig(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialize()
    
    def initialize(self):
        ##### Basic Setup
        self.parser.add_argument("--cuda",    type=str,      help="Cuda ID.")
        self.parser.add_argument("--config",  type=str,      help="Config file Path.", default="./configs/diligent/reading.yml")
        self.parser.add_argument("--exp_code", type=str,      default="exp")

        ##### Testing
        self.parser.add_argument("--testing",       type=str2bool, default=False)
        self.parser.add_argument("--quick_testing", type=str2bool, default=False)
    
    def parse(self):
        args = self.parser.parse_args()
        # Read config file.
        args.config = os.path.expanduser(args.config)
        with open(args.config, "r") as f:
            cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
            cfg = CfgNode(cfg_dict)
        
        if args.quick_testing:
            args.testing = True
        return args, cfg