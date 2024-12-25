import json
import os
import atexit
import requests
import sys
from tqdm import tqdm
import openai
from termcolor import colored
import time
import copy
from utils import read_yaml_file, remove_punctuation
from metrics import *
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

def extract_seconds(text, retried=5):
    words = text.split()
    for i, word in enumerate(words):
        if "second" in word:
            return int(words[i - 1])
    return 60

def strategy_prompt(task, prompt):
    prompt =f"""
    In the task of "{prompt}"

    You need to design strategy about how to change the text to make the task have worse result following the guidence: 
    1. {task}
    2. Keep the semantics the same

    Output the strategy without anything else.
"""
    return prompt


def build_prompt(data, template, type_, few_shot_example = None):

    original_input = "The original text is: "

    original_input = original_input + data + '\n'

    
    attack_objective = " Your task is to generate a new text by the guidence of the following: \n"

    attack_objective += "1. {} \n".format(template)
    attack_objective += "2. Make the LLM get the bad task performance \n"
    attack_objective += "3. keeping the meaning of the original sentences\n"

    if few_shot_example is not None:
        attack_objective += "Here are five examples that fit the guidance: \n"
        for i in range(len(few_shot_example)):
            attack_objective += "{} -> {}\n".format(
                few_shot_example[i][0], few_shot_example[i][1]
            )
    attack_objective += "Only output the new text without anything else."

    prompt = original_input + attack_objective + "\n"
    return prompt


def attack_res(args, instruct, data, type, task, config=None, tokenizer = None, model = None):
    if task == 'cls':
        request_data = form_request(instruct, type, num=1, **config)
        response = openai.ChatCompletion.create(**request_data)
        result = response['choices'][0]['message']['content']
        token = response['usage']['total_tokens']

    else:
        if args.llm_type== 'gpt' or args.llm_type == 'turbo':
            prompt = instruct + ' ' + 'Only output the answer without anything else.'
            system_message = {"role": "system", "content": prompt}
            request_data = form_request(data, type, num=1, system_prompt=system_message, **config)
            response = openai.ChatCompletion.create(**request_data)
            result = response['choices'][0]['message']['content']
            token = response['usage']['total_tokens']
        else:
            raise ValueError('Please change the language model in list.')

    return result, token

def cls_prompt(args, data, example = None):
    prompts = "./data/cls/prompt_ini.json"
    prompts = json.load(open(prompts, "r"))
    prompt = prompts[args.task][args.dataset]

    if example is not None:
        exam = "./data/cls/template_cls.json"
        exam = json.load(open(exam, "r"))
        demons = exam['cls'][args.dataset][1]
        prompt = prompt + '\n' + demons
    prompt = prompt + '\n' + data
    return prompt


def calculate_semantic_similarity(sentence1, sentence2):
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode([sentence1, sentence2])
    
    similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity_score



def cls_select_attack(args, result, data, ref, type, task, config=None, example = None):
    rank = []
    if args.language_model == 'gpt':
        for i in range(len(result)):
            if "turbo" in type or 'gpt4' in type:
                mid_data = result[i]['message']['content']
                prompt = cls_prompt(args, mid_data, example=example)
            else:
                mid_data = result[i]['message']["text"]
                mid_data = mid_data.strip()
                prompt = cls_prompt(args, mid_data, example=example)

            if args.attack_type == 'mix':
                percent = calculate_similarity(mid_data, data)
            elif args.attack_type == 'combine':
                percent = calculate_semantic_similarity(mid_data, data)


            rank.append((i, percent))
        if len(rank) > 0:
            rank_sorted = sorted(rank, key=lambda x: x[1], reverse=True)
            idx = rank_sorted[0][0]
            if "turbo" in type or 'gpt4' in type:
                mid_data = result[idx]['message']['content']
            else:
                mid_data = result[idx]['message']["text"]
                mid_data = mid_data.strip()
            ans = mid_data
            score = rank[0][1]
        else:
            ans = None
            score = rank[0][1]
    return ans, score

def select_attack(args, result, data, raw_data, ref, type, task, config=None, tokenizer = None, model = None, instruct = None, key = None):
    rank = []
    logs = []
    if args.language_model == 'gpt':
        for i in range(len(result)):
            if "turbo" in type or 'gpt4' in type:
                mid_data = result[i]['message']['content']
                if args.task == 'sum':
                    mid_data = data.replace('<change>.', mid_data)
                else:
                    mid_data = mid_data
            else:
                mid_data = result[i]['message']["text"]
                mid_data = mid_data.strip()
                if args.task == 'sum':
                    mid_data = data.replace('<change>.', mid_data)
                else:
                    mid_data = mid_data
            response, token = attack_res(args, instruct, mid_data, type, task, config)
            if args.task == 'sum':
                rouge1, rouge2, rougel = cal_rouge(response, ref)
                score = np.mean([rouge1, rouge2, rougel])
            elif args.task == 'sim':
                score = cal_sari(data, response, ref)
            rank.append((i,score))
        rank_sorted = sorted(rank, key=lambda x: x[1], reverse=False)
        # print(rank_sorted)
        if args.attack_type == 'mix':
            ans = None
            for i in range(len(rank_sorted)):
                index = rank_sorted[i][0]
                if "turbo" in type or 'gpt4' in type:
                    mid_data = result[index]['message']['content']
                else:
                    mid_data = result[index]['message']["text"]
                    mid_data = mid_data.strip()

                percent = calculate_similarity(mid_data, raw_data)
                # print(percent)
                if percent > 0.6:
                    ans = mid_data
                    l_score = rank_sorted[index][1]
                    break
                else:
                    ans = None
                    l_score = None
                    continue

        elif args.attack_type == 'combine':
            for i in range(len(rank_sorted)):
                idx = rank_sorted[i][0]
                if "turbo" in type or 'gpt4' in type:
                    mid_data = result[idx]['message']['content']
                else:
                    mid_data = result[idx]['message']["text"]
                    mid_data = mid_data.strip()

                percent = calculate_semantic_similarity(mid_data, raw_data)

                if percent > 0.8:
                    ans = mid_data
                    l_score = rank_sorted[idx][1]
                    break
                else:
                    ans = None
                    l_score = None
                    continue
        else:
            raise ValueError('No such attack type')
    
    else:
        for idx, text in enumerate(result):
            response = attack_res(args, instruct, text, type, tokenizer=tokenizer, model= model)
            if args.task == 'sum':
                rouge1, rouge2, rougel = cal_rouge(response, ref)
                score = np.mean([rouge1, rouge2, rougel])
            elif args.task == 'sim':
                score = cal_sari(data, response, ref)
            rank.append((idx,score))
        rank_sorted = sorted(rank, key=lambda x: x[1], reverse=False)
        if args.attack_type == 'mix':
            ans = None
            index = rank_sorted[0][0]
            if "turbo" in type or 'gpt4' in type:
                mid_data = result[index]['message']['content']
            else:
                mid_data = result[index]['message']["text"]
                mid_data = mid_data.strip()
            percent = calculate_similarity(mid_data, data)
            ans = mid_data
            l_score = rank_sorted[0][1]

        elif args.attack_type == 'combine':
            idx = rank_sorted[0][0]
            ans = result[idx]
        else:
            raise ValueError('No such attack type')

    return ans, l_score, token

def form_request(data, type, num = 5, system_prompt = None, **kwargs):
    if "davinci" in type:
        request_data = {
            "prompt": data,
            "max_tokens": 1000,
            "top_p": 1,
            "n": num,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stream": False,
            "logprobs": None,
            "stop": None,
            **kwargs,
        }
    else:

        if system_prompt is not None:
            messages_list = [system_prompt]
            messages_list.append({"role": "user", "content": data})
            request_data = {
                "messages": messages_list,
                "max_tokens": 1000,
                "top_p": 1,
                "temperature": 0,
                "stop": None,
                "logprobs": None,
                "n": num,
                **kwargs,
            }
        else:
            messages_list = []
            messages_list.append({"role": "user", "content": data})
            request_data = {
                "messages": messages_list,
                "max_tokens": 1000,
                "top_p": 0.95,
                "temperature": 1,
                "stop": None,
                "logprobs": None,
                "n": num,
                **kwargs,
            }

    return request_data

def llm_init(auth_file="../auth.yaml", llm_type='turbo', setting="default"):
    auth = read_yaml_file(auth_file)[llm_type][setting]
    try:
        openai.api_type = auth['api_type']
        openai.api_base = auth["api_base"]
        openai.api_version = auth["api_version"]
    except:
        pass
    openai.api_key = auth["api_key"]
    return auth



def llm_query(args, prompt, num, is_eval = False, **config):
    model_name = "davinci" if "davinci" in args.llm_type else "turbo"
    request_data = form_request(prompt, model_name, num=num, **config)
    if is_eval is False:
        retried = 0
        while True:
            try:
                if "turbo" in args.llm_type or 'gpt4' in args.llm_type:
                    if num == 1:
                        response = openai.ChatCompletion.create(**request_data)
                        result = response['choices'][0]['message']['content']
                        token = response['usage']['total_tokens']
                    else:
                        response = openai.ChatCompletion.create(**request_data)
                        result = response["choices"]
                        token = response['usage']['total_tokens']
                else:
                    response = openai.Completion.create(**request_data)
                    result = response["choices"]
                break
            except Exception as e:
                error = str(e)
                print("retring...", error)
                second = extract_seconds(error, retried)
                retried = retried + 1
                time.sleep(second)

        # if task:
        # results = [str(r).strip().split("\n\n")[0] for r in result]
        # else:
        #     results = [str(r).strip() for r in res]
    return result, token

def paraphrase(sentence, client, type, **kwargs):
    if isinstance(sentence, list):
        resample_template = [
            f"Generate a variation of the following instruction while keeping the semantic meaning.\nInput:{s}\nOutput:"
            for s in sentence
        ]

    else:
        resample_template = f"Generate a variation of the following instruction while keeping the semantic meaning.\nInput:{sentence}\nOutput:"
    # print(resample_template)
    results = llm_query(resample_template, client, type, False, **kwargs)
    return results


def llm_cls(dataset, client=None, type=None, **config):
    hypos = []
    results = llm_query(dataset, client=client, type=type, task=True, **config)
    if isinstance(results, str):
        results = [results]
    hypos = [remove_punctuation(r.lower()) for r in results]

    return hypos


