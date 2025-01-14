import json
import os
import numpy as np
import heapq
import random

from utils import (
    read_lines,
    open_trans
)
# from llm_att import *
from llm_ge import *
from adversarial_attack import *
from generate_prompt import *
import logging
from transformers import LlamaTokenizer, LlamaForCausalLM



class Adver_Optimaize:
    def __init__(self, args):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.language_model == 'gpt':
            self.model = None
            self.tokenizer = None

        else:
            self.tokenizer = LlamaTokenizer.from_pretrained(args.language_model)
            self.model = LlamaForCausalLM.from_pretrained(args.language_model)
            self.model = self.model.to(self.device)
            self.model.eval()


        self.args = args

        if self.args.prompt is not None:
            self.instruction = self.args.prompt
        else:
            template_file = "./data/cls/prompt_ini.json"
            templates = json.load(open(template_file, "r"))
            self.instruction = templates['cls'][self.args.dataset]
    
    def evaluate_prompts(self, prompt, att):

        prompts = f"""
        {prompt},

        The text is: {att}.

        Only output the answer without anything else.
        """
        
        return prompts
    def white_evaluate(self, prompts, attack_class):
        scores = []
        if self.args.task == 'sum':
            src = './data/xsum/test/attack_res_{}.txt'.format(attack_class)  
            tgt = './data/xsum/test_target/valid.tgt'


            test_src, test_tgt = read_lines(src), read_lines(tgt)  
            test_src = test_src
            test_tgt = test_tgt

            for pro in prompts:
                results = []
                for test in tqdm(test_src):
                    instruction = f"{pro}:\n\n'{test}'"
                    inputs = self.tokenizer(instruction, return_tensors="pt")
                    inputs = {key: value.to(self.device) for key, value in inputs.items()}


                    with torch.no_grad():
                        outputs = self.model.generate(inputs['input_ids'], max_length=5000)

                    output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    ans = output_text.split("\n\n")[-1]
                    print(ans)
                    results.append(ans)

                rouge1, rouge2, rougel = cal_rouge(results, test_tgt)
                score = [rouge1, rouge2, rougel]
                scores.append(score)
                return scores
                    
        elif self.args.task == 'sim':
            src = './data/sim/asset/test_text/attack_res_{}.txt'.format(attack_class)  
            dev_tgt_files = [f'./data/sim/asset/test/asset.valid.simp.{i}' for i in range(10)]
            test_src  = read_lines(src) 
            tgt = []
            for tgt_file in dev_tgt_files:
                tgt.append(read_lines(tgt_file))
            test_tgt = list(map(list, zip(*tgt)))

            test_src = test_src
            test_tgt = test_tgt

            for pro in prompts:
                scores = []
                for test , tgt in tqdm(list(zip(test_src,test_tgt))):
                    instruction = f"{pro}:\n\n'{test}'"
                    inputs = self.tokenizer(instruction, return_tensors="pt")
                    inputs = {key: value.to(self.device) for key, value in inputs.items()}

                    with torch.no_grad():
                        outputs = self.model.generate(inputs['input_ids'], max_length=500)

                    output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    ans = output_text.split("\n\n")[-1]
                    if ':' in ans:
                        ans = ans.split(':')[-1]

                    score = cal_sari(test, ans, tgt, makeadv=True)
                    scores.append(score)

            score = np.mean(scores)
            return score

        elif self.args.task == 'cls':
            self.generate.load_cls_data(self.args)
            self.generate.load_cls_data(self.args, raw_src = False, attack = attack_class, process = 'test')
            src = self.generate.dev_src
            tgt = self.generate.dev_tgt
            att = self.generate.att_src
            template_file = "./data/cls/template_cls.json"
            templates = json.load(open(template_file, "r"))
            prompt = templates['instruction']['opt'].replace('<prompt>', prompts)
            template = templates['cls'][self.args.dataset][1]

            pred = []
            for test in tqdm(att):
                instruction = f"{prompt}\n\n'{template}"
                input = instruction.replace('<input>', test)
                inputs = self.tokenizer(input, return_tensors="pt")
                
                inputs = {key: value.to(self.device) for key, value in inputs.items()}

                with torch.no_grad():
                    outputs = self.model.generate(inputs['input_ids'], max_length=700)

                output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                start_index = output_text.rfind(test)
                
                end_index = output_text.find("\n\n", start_index)
                ans = output_text[start_index:end_index+1].strip()

                pre = open_trans(self.args.dataset, ans)

                pred.append(pre)


            hypos = list(
                    map(
                        lambda x: first_appear_pred(
                            x, dataset_classes_list[self.args.dataset]
                        ),
                        pred,
                    ))
            not_hit = 0
            for i in hypos:
                if i not in dataset_classes_list[self.args.dataset]:
                    not_hit += 1

            score = cal_cls_score(hypos, tgt, metric="acc")

            return score


    
    def black_evaluate(self, prompts, attack_type = None, attack_class = None, judge = True):

        scores = []
        if self.args.task == 'sum':
            if judge:
                src = './data/sum/xsum/test/attack_res_{}.txt'.format(attack_class)  
            else:
                src = './data/sum/xsum/test_target/valid.src'
            tgt = './data/sum/xsum/test_target/valid.tgt'
            print(src)

            test_src, test_tgt = read_lines(src), read_lines(tgt)  


            for pro in prompts:
                results = []
                prompt = pro + ' ' + 'Only output the answer without anything else.'
                system_message = {"role": "system", "content": prompt}

                
            for batch_prompt in tqdm(test_src):
                ans = llm_query(batch_prompt, self.args.llm_type, num = 1, system_prompt=system_message, geneprompt=True, **self.generate.llm_config)
                results.append(ans)

            rouge1, rouge2, rougel = cal_rouge(results, test_tgt)
            score = [rouge1, rouge2, rougel]
            scores.append(score)
            return scores
        
        elif self.args.task == 'sim':
            if judge:
                src = './data/sim/asset/test_text/attack_res_{}.txt'.format(attack_class)  
            else:
                src = './data/sim/asset/test/asset.valid.src'
            dev_tgt_files = [f'./data/sim/asset/test/asset.valid.simp.{i}' for i in range(10)]
            test_src  = read_lines(src) 

            tgt = []
            for tgt_file in dev_tgt_files:
                tgt.append(read_lines(tgt_file))
            test_tgt = list(map(list, zip(*tgt)))


            scores = []
            for pro in prompts:
                results = []
                prompt = pro + ' ' + 'Only output the answer without anything else.'
                system_message = {"role": "system", "content": prompt}
                for batch_prompt, tgt in tqdm(list(zip(test_src,test_tgt))):
                    ans  = llm_query(batch_prompt, self.args.llm_type, num = 1, system_prompt=system_message, geneprompt=False, **self.generate.llm_config)
                    score = cal_sari(batch_prompt, ans, tgt, makeadv=True)
                    scores.append(score)

            score = np.mean(scores)
            return score
        
        elif self.args.task == 'cls':
            self.generate.load_cls_data(self.args)
            self.generate.load_cls_data(self.args, raw_src = False, attack = attack_class, process = 'test')
            src = self.generate.dev_src
            tgt = self.generate.dev_tgt
            att = self.generate.att_src


            for prompt in prompts:
                pred = []
                system_message = {"role": "system", "content": prompt}

                for data in tqdm(att):
                    pre  = llm_query(data, self.args.llm_type, num=1, system_prompt=system_message, **self.generate.llm_config)
                    pred.append(pre)
                hypos = list(
                        map(
                            lambda x: first_appear_pred(
                                x, dataset_classes_list[self.args.dataset]
                            ),
                            pred,
                        ))
                not_hit = 0
                for i in hypos:
                    if i not in dataset_classes_list[self.args.dataset]:
                        not_hit += 1
                # print(f"ratio: {not_hit/len(hypos)}")
                score = cal_cls_score(hypos, tgt, metric="acc")

            return score
        


    def forward(self):
        raise NotImplementedError



class Adversarial(Adver_Optimaize):
    def __init__(self, args, attack):
        super(Adversarial, self).__init__(args)
        self.args = args
        self.attack = attack

    def forward(self, datagen = True, weak = None, instruction = None):
        if instruction is not None:
            self.instruction = instruction

        if self.args.sample_num > len(self.attack.dev_src):
            raise ValueError("The sample number must less than the length of dataset")

        if self.args.task == 'cls':
            if datagen is True:
                src = self.attack.dev_src
                tgt = self.attack.dev_tgt

                for src, tgt in tqdm(list(zip(self.attack.dev_src, self.attack.dev_tgt))):
                        answer = self.attack.forward(src, tgt, self.instruction, genedata=True, weak=weak)
                        for idx, ans in enumerate(answer):
                            if ans == src:
                                with open('./data/cls/{}/attack_res_{}.txt'.format(self.args.dataset, idx), 'a', encoding='utf-8') as f:
                                    f.write("Need to Regenerate" + '\n')

                            else:
                                with open('./data/cls/{}/attack_res_{}.txt'.format(self.args.dataset, idx), 'a', encoding='utf-8') as f:
                                    f.write(ans + '\t' + tgt + '\n')

  
            else:
                answer = []
                if self.args.attack_type == 'mix':
                    total = 0
                    random.seed(time.time())
                    print("****Adversarial Attack*****") 
                    idx =  random.sample(range(0, len(self.attack.dev_src)+1), self.args.sample_num) 
                    for i in idx:
                        src, tgt = self.attack.dev_src[i], self.attack.dev_tgt[i]
                        ans  = self.attack.forward(src, tgt, self.instruction, genedata = False, weak = weak)
                        answer.append(ans)
                        # total += token
                elif self.args.attack_type == 'combine':
                    random.seed(time.time())
                    total = 0
                    print("****Adversarial Attack*****") 
                    idx =  random.sample(range(0, len(self.attack.dev_src)+1), self.args.sample_num) 
                    for i in idx:
                        src, tgt = self.attack.dev_src[i], self.attack.dev_tgt[i]
                        ans  = self.attack.forward(src, tgt, self.instruction, genedata = False, weak = weak)
                        answer.append(ans)
                        # total += token

                return idx, answer

                                
        else:
            if datagen is True:
                print('begin generate data!!!!')

                for src, tgt in tqdm(list(zip(self.attack.dev_src, self.attack.dev_tgt))):
                    answer, token = self.attack.forward(src, tgt, self.instruction)

                    for idx, ans in enumerate(answer):
                        if self.args.task == 'sum':
                            with open('./data/sum/xsum/train/attack_res_{}.txt'.format(idx), 'a', encoding='utf-8') as f:
                                if ans == src:
                                    f.write("Need to Regenerate"+ '\n')
                                else:
                                    ans = ans.replace('\n','')
                                    f.write(ans+ '\n')
                        else :
                            with open('./data/sim/asset/attack_res_{}.txt'.format(idx), 'a', encoding='utf-8') as f:
                                if ans == src:
                                    f.write("Need to Regenerate"+ '\n')
                                else:
                                    ans = ans.replace('\n','')
                                    f.write(ans+ '\n')
                            # with open('./data/xsum/score_2.txt'.format(idx+1), 'a', encoding='utf-8') as f:
                            #     f.write(str(score[0]) + ' ' + str(score[1]) + ' ' + str(score[2]) + '\n')
                            # logging.info('One result has been wirtten')

                    # else:
                    #     with open('./data/xsum/attack_res_{}.txt'.format(self.args.task), 'a', encoding='utf-8') as f:
                    #         f.write(answer + '\n')
                    #         logging.info('One result has been wirtten')
            else:
                answer = []
                if self.args.attack_type == 'mix':
                    total = 0
                    print("****Select the Sample*****")
                    random.seed(time.time())
                    idx =  random.sample(range(0, len(self.attack.dev_src) + 1), self.args.sample_num) 

                    print("****Adversarial Attack*****")
                    for i in tqdm(idx):
                        src, tgt = self.attack.dev_src[i], self.attack.dev_tgt[i]
                        ans, token = self.attack.forward(src, tgt, self.instruction, genedata = False, weak = weak)
                        answer.append(ans)
                        total += token
                elif self.args.attack_type == 'combine':
                    print("****Select the Sample*****")
                    random.seed(time.time())
                    total = 0
                    idx =  random.sample(range(0, len(self.attack.dev_src) + 1), self.args.sample_num)

                    print("****Adversarial Attack*****") 
                    for i in idx:
                        src, tgt = self.attack.dev_src[i], self.attack.dev_tgt[i]
                        ans, token = self.attack.forward(src, tgt, self.instruction, genedata = False, weak = weak)
                        answer.append(ans)
                        total += token

                return idx, answer


class Open_optimize(Adver_Optimaize):
    def __init__(self, args, generate):
        super(Open_optimize, self).__init__(args)
        self.args = args
        self.generate = generate
        template_file = "./temp_file/attack_temp.json"
        templates = json.load(open(template_file, "r"))
        temp = templates[self.args.attack_type]
        self.iters = len(temp)
        self.llm_config = llm_init(f"./temp_file/auth.yaml", args.llm_type, args.setting)

    
    def open_evaluate(self, prompt, src, attack, tgt):
        
        tokens = 0
        if self.args.task == 'sum':
            result_att = []
            for att in tqdm(attack):
                prompt_att = self.evaluate_prompts(prompt, att)
                # prompt_nor = self.evaluate_prompts(prompt, src[num])
                result = llm_query(prompt_att, self.args.llm_type, **self.generate.llm_config)
                # tokens += token
                # result_nor = llm_query(prompt_nor, self.args.llm_type, **self.generate.llm_config)
                result_att.append(result)
            rouge1, rouge2, rougel = cal_rouge(result_att, tgt)
            score_att = np.mean([rouge1, rouge2, rougel])

            score = score_att

        elif self.args.task == 'sim':
            scores = []
            for idx, att in enumerate(tqdm(attack)):
                prompt_att = self.evaluate_prompts(prompt, att)
                result = llm_query(prompt_att, self.args.llm_type, **self.generate.llm_config)
                # tokens += token
                scores.append(cal_sari(src[idx], result, self.generate.dev_tgt[idx], makeadv=True))
            # score_nor = cal_sari(src, result_nor, self.generate.dev_tgt)
            score = np.mean(scores)
        
        return score
    def cls_evaluate(self, prompt, src, tgt):
        system_message = {"role": "system", "content": prompt}
        pred = []
        tokens = 0
        for data in tqdm(src):
            pre = llm_query(data, self.args.llm_type, num=1, system_prompt=system_message, **self.generate.llm_config)
            # tokens += token
            pred.append(pre)

        hypos = list(
                map(
                    lambda x: first_appear_pred(
                        x, dataset_classes_list[self.args.dataset]
                    ),
                    pred,
                )
            )
        not_hit = 0
        for i in hypos:
            if i not in dataset_classes_list[self.args.dataset]:
                not_hit += 1
        print(f"ratio: {not_hit/len(hypos)}")
        score = cal_cls_score(hypos, tgt, metric="acc")
        return score
    def load_att_data(self, i = None):
        if self.args.task == 'sum':
            if i is None:
                path = './data/sum/xsum/data_enhance.txt'
            else:
                path = './data/sum/xsum/train/attack_res_{}.txt'.format(i)
            att_src = read_lines(path)
        elif self.args.task == 'sim':
            if i is None:
                path = './data/sim/asset/data_enhance.txt'
            else:
                path = './data/sim/asset/train/attack_res_{}.txt'.format(i)
            att_src = read_lines(path)
        
        return att_src

    
    def forward(self, idx, weak, att=None, instruction = None):
        if instruction is None:
            instructions = self.instruction
        else:
            instructions = instruction

        if self.args.task == 'sum':
            self.generate.load_sum_data(self.args)

            src = self.generate.dev_src
            tgt = self.generate.dev_tgt
        
        elif self.args.task == 'sim':
            self.generate.load_sim_data(self.args, raw_data = False)
            # sample_indice = random.sample(len(self.generate.dev_att), 1)
            att = att
            src = self.generate.dev_src
            tgt = self.generate.dev_tgt

            # tgt = []
            # for i in self.generate.dev_tgt:
            #     tgt.append([i[idx]])
        elif self.args.task == 'cls':
            
            self.generate.load_cls_data(self.args)
            src = self.generate.dev_src
            tgt = self.generate.dev_tgt

        results = []
        total = 0

    
        instruction = instructions


        src_data = []
        tgt_data = []
        prompt_dict = []
        example_att = "Here are some examples of perturbations:"

        for i, id in enumerate(idx):
            src_data.append(src[id])
            tgt_data.append(tgt[id])
            example_att += format_template(src = src[id], tgt = att[i])
            example_att += '\n'
            
        prompts= self.generate.forward(self.args, src_data, att, tgt_data, example_att, instruction, weak)

    
        # prompts = data_enhance(instruction, self.args.llm_type, self.args.task, **self.llm_config)

        if self.args.task in ['sim', 'sum']:
            for index,prompt in enumerate(tqdm(prompts)):
                # print(prompt)
                    random.seed(time.time())
                    indices = list(range(len(tgt)))
                    selected_indices = random.sample(indices, 20)
                    src_cho = [src[i] for i in selected_indices]
                    tgt_cho = [tgt[i] for i in selected_indices]
                    attack = self.load_att_data()
                    scores = []

                    for key, value in weak.items():
                        attack = self.load_att_data(key)
                        att_cho = [attack[i] for i in selected_indices]
                        score = self.open_evaluate(prompt, src_cho, att_cho, tgt_cho)
                        scores.append(score)

                    
                    score = np.mean(scores)
                    prompt_dict.append((index, score))

            sorted_prompt_dict = sorted(prompt_dict, key=lambda x: x[1], reverse=True)
            results.extend([prompts[item[0]] for item in sorted_prompt_dict[:5]])

        
        elif self.args.task == 'cls':
            for index,prompt in enumerate(tqdm(prompts)):
                scores = []
                random.seed(time.time())
                indices = list(range(len(tgt)))
                selected_indices = random.sample(indices, 20)

                # self.generate.load_cls_data(self.args, raw_src = False)
                # att_cho, tgt_cho = zip(*[(self.generate.att_src[i], tgt[i]) for i in selected_indices])
                # score = self.cls_evaluate(prompt, att_cho, tgt_cho)

                for key, value in weak.items():
                    self.generate.load_cls_data(self.args, raw_src = False, attack = key)
                    att_cho, tgt_cho = zip(*[(self.generate.att_src[i], tgt[i]) for i in selected_indices])
                    score = self.cls_evaluate(prompt, att_cho, tgt_cho)
                    scores.append(score)
                score = np.mean(scores)
                prompt_dict.append((index, score))

            sorted_prompt_dict = sorted(prompt_dict, key=lambda x: x[1], reverse=True)
            results.extend([prompts[item[0]] for item in sorted_prompt_dict[:5]])

        return results