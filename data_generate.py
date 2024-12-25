import json
import os
from tqdm import tqdm
import numpy as np
import random
import sys
import time
import argparse
from args import parse_args
from utils import *
from llm_ge import *
from metrics import *
from adversarial_attack import Opentask
from generate_prompt import Opentask_Ge
from process import Adversarial, Open_optimize
import logging
import json



def train(args):
    set_seed(args.seed)
    advattack = Opentask(args)


    attack = advattack
    adv = Adversarial(args, attack=attack)


    weak_file = "./temp_file/weak.json"
    weak_type = json.load(open(weak_file, "r"))
    weak = weak_type[args.task][args.attack_type]


    adv.forward(datagen=True, weak = weak, instruction=args.prompt)


if __name__ == '__main__':

    parser = parse_args()
    args = parse_args()
    parser.add_argument('--use_data', type=bool, default = False)
    parser.add_argument('--data_path_train', type=str)
    parser.add_argument('--data_path_test', type=str)
    args = parser.parse_args()

    train(args)