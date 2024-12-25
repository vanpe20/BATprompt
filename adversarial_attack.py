import json
import os
from tqdm import tqdm
import numpy as np
import random
import sys
import time
from utils import *
from llm_att import *
from metrics import *
import logging

class Adv_attack(object):
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = args.dataset
        template_file = "./temp_file/attack_temp.json"
        templates = json.load(open(template_file, "r"))

        if 'mix' in args.attack_type:
            a_type = 'mix'
        elif 'combine' in args.attack_type:
            a_type = 'combine'
        else:
            raise ValueError("the attack type must choose from mix and combine")
        
        self.template = templates[a_type] 
        
        self.client = None
        self.llm_config = llm_init(f"./temp_file/auth.yaml", args.llm_type, args.setting)

    
    def attack_generation(self, eval_src, ref_texts, instruct, genedata = True, weak = None):
        tokens = 0
        args = self.args
        if weak is not None:
            self.template = weak


        results = []
        combine_attack = []
        final_score = []
        if 'turbo' in args.llm_type or 'gpt' in args.llm_type:
            if args.task == 'cls':
                data, ref = eval_src, ref_texts
                inp = data
                for key, value in self.template.items():
                    if key == 'C1':
                        prompt = strategy_prompt(instruct, value)
                        strategy = llm_query(args, prompt, 1, **self.llm_config)
                    else:
                        strategy = value
                    if genedata is True and args.attack_type == 'mix':
                        inp = data
                    h_score = 100
                    score_f = None
                    for i in range(self.args.iter):
                        prompt = build_prompt(inp, strategy, args.task)
                        attack_answer, token = llm_query(args, prompt, 5, **self.llm_config)
                        tokens += token
                        ans, score= cls_select_attack(args, attack_answer, data, ref, args.llm_type, args.task, self.llm_config)

                        if ans is not None and score < h_score :
                            inp = ans
                            h_score = score
                        else:
                            inp = inp

                    if genedata is True or args.attack_type == 'combine':
                        combine_attack.append(inp)
                        final_score.append(score_f) 
                if genedata is False and args.attack_type == 'mix':
                    results = inp
                else:
                    results = combine_attack
                return results
                    
            else:
                dataset, ref_text = eval_src, ref_texts
                for key, value in self.template.items():

                    if key == 'C1':
                        prompt = strategy_prompt(instruct, value)
                        strategy = llm_query(args, prompt, 1, **self.llm_config)
                    else:
                        strategy = value
                    lowst_score = 100
                    data = dataset
                    wor_data = data
                    for i in range(self.args.iter):

                        if args.task == 'sum':
                            sentences = data.split('.')
                            sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
                            random.seed(time.time())

                            if len(sentences) > 2:
                                index = random.randint(0, len(sentences) - 2)
                                inp = sentences[index] + '. ' + sentences[index + 1]
                                sentences[index] = "<change>"
                                del sentences[index + 1] 
                                data = ". ".join(sentences) + "."
                            else:
                                inp = data
                        else:
                            inp = data
                        prompt = build_prompt(inp, strategy, args.task)
                        attack_answer, token= llm_query(args, prompt, 5, **self.llm_config)
                        tokens += token
                        ans, l_score, token = select_attack(args, attack_answer, data, inp, ref_text, args.llm_type, args.task, self.llm_config, instruct=instruct, key = key)
                        tokens +=token
                        if ans is not None and l_score < lowst_score:
                            if args.task == 'sum':
                                data = data.replace('<change>.', ans)
                            else:
                                data = ans
                            wor_data = data
                            lowst_score = l_score
                            data = wor_data
                        else:
                            data = wor_data

                    if genedata is True or args.attack_type == 'combine':
                        combine_attack.append(wor_data)

                if genedata is False and args.attack_type == 'mix':
                    results = wor_data
                else:
                    results = combine_attack
                # print(ssd)
                return results, tokens

        else:
            raise ValueError('Please change language model in llm_type list')

    def eval_attack_generation(self, args, src, ref, attack):
        if isinstance(src, list):
            results = []
            for text in tqdm(src):
                results.append(llm_query(args, src, ref, attack, client = self.client, type=args.llm_type, task=True, instruct = None, is_eval=True))
        
        return results


class Opentask(Adv_attack):
    def __init__(self, args, use_data = False):
        super(Opentask, self).__init__(args)
        self.args = args
        if args.task == 'sum':
            self.dev_src, self.dev_tgt, self.test_src, self.test_tgt = load_sum_data(
                args, args.dataset, args.seed, use_data = use_data
            )
        elif args.task == 'sim':
            
            self.dev_src, self.dev_tgt, self.test_src, self.test_tgt = load_sim_data(
                args, args.dataset, args.seed
            )
        
        elif args.task == 'cls':
            self.verbalizers = get_dataset_verbalizers(args.dataset)
            data_store_path = './data/cls/{}/dev_raw.txt'.format(args.dataset)
            data_store = read_lines(data_store_path)
            self.dev_src = [line.split("\t")[0] for line in data_store]
            self.dev_tgt = [
                self.verbalizers[int(line.strip().split("\t")[1])]
                for line in data_store]


    def forward(
        self,  eval_src=None, ref_texts=None, instruct = None, attack=None, eval=False, genedata=True, weak = None
    ):  
        if eval is False:
            answer = self.attack_generation(eval_src, ref_texts, instruct, genedata=genedata, weak = weak)
        else:
            answer = self.eval_attack_generation(self.args, eval_src, ref_texts, attack)
        return answer
