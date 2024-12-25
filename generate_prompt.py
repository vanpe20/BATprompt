import json
import os
from tqdm import tqdm
import numpy as np
import random
import sys
import time
from utils import *
from llm_ge import *
from metrics import *


class Generate(object):
    def __init__(self,args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = args.dataset


        dataset = self.dataset


        self.llm_config = llm_init(f"./temp_file/auth.yaml", args.llm_type, args.setting)

    def prompt_generation(self, args, source_text, attack_text, ref_text, demons, raw_instruction, weak):
        tokens = 0
        prompts = build_prompt_diff(demons, weak, args.attack_type)
        if isinstance(prompts, list):
            prompt_diff = ''
            for prompt in prompts:
                diff  = llm_query(prompt, args.llm_type, num=1, **self.llm_config)
                # tokens += token
                prompt_diff += diff
                prompt_diff +='\n'
        else:
            prompt_diff = llm_query(prompts, args.llm_type, num=1, **self.llm_config)
            # tokens += token
        prompt_ge = build_prompt_gen(prompt_diff, args.attack_type)
            
        guide =  llm_query(prompt_ge, args.llm_type, num=1, **self.llm_config)
        # tokens +=token
        prompt = build_prompt_new(raw_instruction, guide, args.task)
        results, token =  gene_prompt(prompt, args.llm_type, **self.llm_config)
        # tokens +=token
  
        ans = paraphrase(args, results, args.llm_type, **self.llm_config)
        results += ans
        # tokens +=token

        return results

class Opentask_Ge(Generate):
    def __init__(self, args):
        super(Opentask_Ge, self).__init__(args)

    def load_sum_data(self, args):
        self.dev_src, self.dev_tgt, self.test_src, self.test_tgt = load_sum_data(
                args, args.dataset, args.seed)


    def load_sim_data(self, args, raw_data=True):

        if raw_data is True:
            self.dev_src, self.dev_tgt, self.test_src, self.test_tgt = load_sim_data(
                args, args.dataset, args.seed, raw_data=raw_data
            )
        else:
            self.dev_src, self.dev_tgt = load_sim_data(
                args, args.dataset, args.seed,  raw_data=raw_data
            )

    def load_cls_data(self, args, raw_src = True, attack = None, process = 'train'):

        if raw_src is True:
            self.verbalizers = get_dataset_verbalizers(args.dataset)
            data_store_path = './data/cls/{}/dev.txt'.format(args.dataset)
            data_store = read_lines(data_store_path)
            self.dev_src = [line.split("\t")[0] for line in data_store]
            self.dev_tgt = [
            self.verbalizers[int(line.strip().split("\t")[1])]
                for line in data_store]
        else:
            data_store_path = './data/cls/{}/{}/attack_res_{}.txt'.format(args.dataset, process, attack)
            data_store = read_lines(data_store_path)
            self.att_src = [line.split('\t')[0] for line in data_store]

        

    def forward(self, args, src_text, attack_text, tgt_text, build_instruction, raw_instruction, weak):
        results = self.prompt_generation(args, src_text, attack_text, tgt_text, build_instruction, raw_instruction, weak)

        return results