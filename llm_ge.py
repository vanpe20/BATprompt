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
from utils import read_yaml_file
from metrics import *
import numpy as np


def extract_seconds(text, retried=5):
    words = text.split()
    for i, word in enumerate(words):
        if "second" in word:
            return int(words[i - 1])
    return 60


def build_prompt_diff(demons, weak, type=None, idx = None, corr = False):
    if corr is False:
        template = 'There are several operations: '
        for key, value in weak.items():
            template += value
            template += '\n'
        template += "After these operations the Original text becomes Perturbed text\n"
        template += "Please combine these operations and compare the following texts and explain how the Perturbed text differs from the Original text.\n Please provide a detailed comparison highlighting key differences. \n"

        template += demons
        guidence= "Only output the differences without anything else."

        prompt = template+ "\n" + guidence + "\n"
    else :
        if type == 'combine':
            prompt = []
            num = len(weak)
            i = 0
            for key, value in weak.items():
                template = 'There is an operations: '
                template += value
                template += '\n'
                template += "After this operation the Original text becomes Perturbed text\n"
                template += "Please combine this operation and compare the following texts and explain how the Perturbed text differs from the Original text.\n Please provide a detailed comparison highlighting key differences. \n"
                for j in range(num):
                    template += demons[i+j*num]
                    template +='\n'
                guidence = "Only output the differences without anything else."
                prompt.append(template + guidence + "\n")
                i +=1
        else:
            raise ValueError('Type Error!')
    return prompt


    # else:
    #     template = "Compare the following two sentences and explain why is the second sentence worse than the first :\n 1. <First>.\n 2.<Second>\n. Please provide a detailed comparison highlighting key differences. \n"

    #     template = template.replace("<First>", src).replace("<Second>", att)
    #     guidence= "Only output the differences without anything else."
    #     prompt = template + guidence + "\n"
    
    # return prompt, value

def build_prompt_gene(instruction, src):
    prompt = instruction.replace("<input>", src)
    return prompt

def build_prompt_mix(instruction, prompt_diff, ans_diff):
    template = " In the task of {}. ".format(instruction)
    diff = "Analyze how the difference in input has casued the difference in answer.\n The input difference is <InDiff>. \b The answer difference is <OutDiff>. "
    diff = diff.replace("<InDiff>", prompt_diff).replace("<OutDiff>", ans_diff)
    guidence = "Only output the analyse without anything else."
    prompt = template + diff + guidence
    return prompt


def build_prompt_com(instruction, prompt_diff, ans_diff):
    template = " In the task of {}. ".format(instruction)
    diff = "There are several pairs of differences, Please analyze how the difference in prompt has casued the difference in answer.\n"
    if isinstance(prompt_diff, list):
        if len(prompt_diff) == len(ans_diff):
            for inp, out in zip(prompt_diff, ans_diff):
                diff += "Prompt Difference: {} \n".format(prompt_diff)
                diff += "Answer Difference: {} \n".format(ans_diff)
                
        else:
            raise ValueError("The length should be same!")
    else:
        diff += "Prompt Difference: {} \n".format(prompt_diff)
        diff += "Answer Difference: {} \n".format(ans_diff)

    guidence = "Only output the analyse without anything else."
    prompt = template + diff + guidence
    return prompt

def build_prompt_new(instruction, cause, task):

    if task in ['sim', 'sum']:
        if task == 'sim':
            word = 'simplification'
        else:
            word = 'summarization'

        template = f"""
            I'm trying to write a zero-shot text-{word} prompt.
        
            My current prompt is:
            "{instruction}"

            But this prompt gets the answer with low performance because the some sentence of the input is changed by following operations:

            {cause}

            Based on the above difference, I wrote 1 different improved prompts to correct these differences and make a better task performance.

            Only output the prompt without anything else.

            The 1 new prompts are:
            """
    elif task == 'cls':
        template = f"""
            I'm trying to write a zero-shot classification tasks prompt.
        
            My current prompt is:
            "{instruction}"

            But this prompt gets the wrong classification because the input is changed a bit from the original text

            the change of the input is : {cause}

            Based on the above change, I wrote 1 different improved prompts to correct these differences.

            Only output the prompt without anything else.

            he 1 new prompts are:
            """

    # template = "Please use the difference {} ,which may have negative influence on final task to fix the prompt {} to generate a better prompt for this task.".format(cause, instruction)


    return template

def build_prompt_gen(cause, type):
    
    if type == 'combine':
        template = f"""
        Here are some difference of two text:
        {cause}

        you should summary general difference about text structure based on it.

        Only output the answer without anything else.
        """

    else:
        template = f"""
        Here are some difference of two contexts:
        
        {cause}

        You should summary a general difference based on it.

        Only output the answer without anything else.
        """
    return template


def form_request(data, type, num = 5, system_prompt = None, geneprompt = True ,**kwargs):

    if system_prompt:
        messages_list = [system_prompt]
    else:   
        messages_list = []
    messages_list.append({"role": "user", "content": data})
    if geneprompt is True:
        request_data = {
            "messages": messages_list,
            "max_tokens": 1000,
            "top_p": 0.95,
            "stop": None,
            "n": num,
            **kwargs,
        }

    else:
        request_data = {
            "messages": messages_list,
            "max_tokens": 1000,
            "top_p": 1,
            "temperature": 0,
            "stop": None,
            "n": num,
            **kwargs,
        }
    # print(request_data)
    return request_data

def llm_init(auth_file="./temp_file/auth.yaml", llm_type='davinci', setting="default"):
    auth = read_yaml_file(auth_file)[llm_type][setting]
    try:
        openai.api_type = auth['api_type']
        openai.api_base = auth["api_base"]
        openai.api_version = auth["api_version"]
    except:
        pass
    openai.api_key = auth["api_key"]
    return auth



def llm_query(prompt, type, num = 1, task = True, system_prompt = None, geneprompt = True, **config):
    results = []
    model_name = "davinci" if "davinci" in type else "turbo"
    request_data = form_request(prompt, model_name, num, system_prompt = system_prompt, geneprompt=geneprompt, **config)
    retried = 0
    if not isinstance(prompt, list):
        while True:
            try:
                if "turbo" in type or 'gpt4' in type:
                    response = openai.ChatCompletion.create(**request_data)
                    if num == 1:
                        results = response["choices"][0]['message']['content']
                    else:
                        for i in range(5):
                            result = response["choices"][i]['message']['content']
                            results.append(result)
                    # token = response['usage']['total_tokens']
                    break
                else:
                    break
            except Exception as e:
                error = str(e)
                print("retring...", error)
                second = extract_seconds(error, retried)
                retried = retried + 1
                time.sleep(second)

    return results

def gene_prompt(prompt, type, **config):
    model_name = type
    request_data = form_request(prompt, model_name,**config)
    retried = 0
    while True:
        try:
            if "turbo" in type or 'gpt4' in type:
                response = openai.ChatCompletion.create(**request_data)
                results = response["choices"]
                token = response['usage']['total_tokens']   
                break
            else:
                break    
        except Exception as e:
            error = str(e)
            print("retring...", error)
            second = extract_seconds(error, retried)
            retried = retried + 1
            time.sleep(second)
    answer = []
    for i in range(len(results)):
        if "turbo" in type or 'gpt4' in type:
            answer.append(results[i]['message']['content'])
        else:
            answer.append(results[i]['message']["text"])
    return answer, token



def data_enhance(sentence, type, task, **kwargs):
    resample_template = f"Paraphrase the following instruction while keeping the semantic meaning and make summary task has better performance.\nInput:{sentence}"
    results= llm_query(resample_template, type, num = 5, **kwargs)

    return results



def paraphrase(args, sentence, type, **kwargs):
    if isinstance(sentence, list):
        resample_template = [
            f"Paraphrase the following instruction while keeping the semantic meaning.\nInput:{s}\nOutput:"
            for s in sentence
        ]

    else:
        resample_template = f"Paraphrase the following instruction while keeping the semantic meaning.\nInput:{sentence}"

    results = llm_query(resample_template, type, num = args.generate, **kwargs)

    return results

