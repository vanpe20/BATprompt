import re
import yaml
import random
import string
import json
import logging
from logging.handlers import TimedRotatingFileHandler
import torch
from typing import List
import numpy as np

dataset_classes_list = {
    'sst2': ['positive', 'negative'],
    'mr': ['positive', 'negative'],
    'cr': ['positive', 'negative'],
    'subj': ['subjective', 'objective'],
    'agnews': ['World', 'Sports', 'Business', 'Tech'],
    'trec': ['Description', 'Entity', 'Expression', 'Human', 'Location', 'Number'],
    'sst-5': ['terrible', 'bad', 'okay', 'good', 'great'],
}

def open_trans(data, ans):
    if data in ['cr','mr', 'sst2']:
        if 'negative' in ans.lower() or 'Negative' in ans.lower():
            return 'Negative'
        elif 'postive' in ans.lower() or 'positive' in ans.lower():
            return 'Postive'
        else:
            return 'No'
    elif data == 'agnews':
        if 'World' in ans:
            return 'World'
        elif 'Sports' in ans:
            return 'Sports'
        elif 'Business' in ans:
            return 'Business'
        elif 'Tech' in ans:
            return 'Tech'
        else:
            return 'No'
    elif data == 'trec':
        if 'description' in ans.lower():
            return 'Description'
        elif 'entity' in ans.lower():
            return 'Entity'
        elif 'expression' in ans.lower():
            return 'Expression'
        elif 'human' in ans.lower():
            return 'Human'
        elif 'location' in ans.lower():
            return 'Location'
        elif 'number' in ans.lower():
            return 'Number'
        else:
            return 'No'

    elif data == 'sst-5':
        if 'terrible' in ans.lower():
            return 'terrible'
        elif 'bad' in ans.lower():
            return 'bad'
        elif 'okay' in ans.lower():
            return 'okay'
        elif 'good' in ans.lower():
            return 'good'
        elif 'great' in ans.lower():
            return 'great'
        else: 
            return 'No'

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def remove_punctuation(s): 
    translator = str.maketrans('', '', string.punctuation)
    return s.translate(translator)

def first_appear_pred(text, verbalizer_dict): 
    text = text.lower()
    verbalizer_dict = [k.lower() for k in verbalizer_dict]
    for word in text.split():
        if word in verbalizer_dict:
            return word

    return ""


def count_lines(file_path):
    with open(file_path, 'r') as f:
        return sum(1 for _ in f)


def read_lines(file_, sample_indices=None):
    ret = []
    if sample_indices:
        sample_indices.sort()
        with open(file_, 'r') as f:
            for i, line in enumerate(f):
                if i in sample_indices:
                    ret.append(line.rstrip())
        return ret
    else:
        with open(file_, 'r') as f:
            lines = f.readlines()
        return [line.rstrip() for line in lines]


def format_template(
    src,
    tgt,
): 
    if isinstance(tgt, list):
        template = ''
        for item in tgt:
            template += f"Original text: {src}"
            template += '\n'
            template += f"Perturbed text: {item}"

    else:
        template = f"""
            Original text: {src}
            Perturbed text: {tgt}
        """

    return template


def load_cls_data(verbalizers=None, data_path=None,  sample_indices=None): 
    test_data = read_lines(
        data_path, sample_indices=sample_indices)
    test_src = []
    test_tgt = []
    for i, line in enumerate(test_data):
        try:
            cur_src, cur_tgt = line.split('\t')
        except:
            raise ValueError
        test_src.append(cur_src)
        test_tgt.append(verbalizers[int(cur_tgt)])
    return test_src, test_tgt


def load_sum_data_(args, src_file, tgt_file, sample_indices=None):
    src = read_lines(src_file, sample_indices=sample_indices)
    tgt = read_lines(tgt_file, sample_indices=sample_indices)


    return src, tgt

def load_sum_data(args, dataset, seed, sample_num = None, use_data = False):
    random.seed(seed)
    if use_data is True:
        dev_file = args.data_path_train
        test_file = args.data_path_test
    if dataset == 'sam':
        dev_file = './data/sum/sam/train_target/valid'
        test_file = './data/sum/sam/test_target/valid'
        dev_src, dev_tgt = load_sum_data_(args, f'{dev_file}.src',f'{dev_file}.tgt')
        test_src, test_tgt = load_sum_data_(args, f'{test_file}.src',f'{test_file}.tgt')
        return dev_src, dev_tgt, test_src, test_tgt



def load_sim_data_(args, src_file, tgt_files, sample_indices=None): 
    src = read_lines(src_file, sample_indices=sample_indices)
    tgt = []
    for tgt_file in tgt_files:
        tgt.append(read_lines(tgt_file, sample_indices=sample_indices))
    tgt = list(map(list, zip(*tgt)))

    return src, tgt



def load_sim_data(args, dataset, seed, raw_data=True):
    random.seed(seed)
    if raw_data is True:
        if dataset == 'asset':
            dev_src_file = './data/sim/asset/dev/asset.valid.src'
            # dev_tgt_file = './data/sim/asset/dev/asset.valid.tgt'
            dev_tgt_files = [
                f'./data/sim/asset/dev/asset.valid.simp.{i}' for i in range(10)]
            test_src_file = './data/sim/asset/test/asset.valid.src'
            # test_tgt_file = './data/sim/asset/dev/asset.test.tgt'
            test_tgt_files = [
                f'./data/sim/asset/test/asset.valid.simp.{i}' for i in range(10)]
        else:
            raise ValueError("dataset not supported")
        test_src, test_tgt = load_sim_data_(args, test_src_file, test_tgt_files)
        dev_src, dev_tgt = load_sim_data_(args, dev_src_file, dev_tgt_files)
        return dev_src, dev_tgt, test_src, test_tgt
    else:
        # dev_att_file = './data/sum/sam/attack_res_sim.txt'
        dev_src_file = './data/sim/asset/dev/asset.valid.src'
        dev_tgt_files = [
                f'./data/sim/asset/dev/asset.valid.simp.{i}' for i in range(10)]

        # dev_att, _ = load_sim_data_(args, dev_att_file, dev_tgt_files)

    dev_src, dev_tgt = load_sim_data_(args, dev_src_file, dev_tgt_files)

    return  dev_src, dev_tgt




def batchify(data, batch_size=16): #将数据分批处理
    batched_data = []
    for i in range(0, len(data), batch_size):
        batched_data.append(data[i:i + batch_size])
    return batched_data

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def get_dataset_verbalizers(dataset: str) -> List[str]:
    
    if dataset in ["sst2",  "mr", "cr"]:
        verbalizers = ["negative", "positive"]  # num_classes
    elif dataset == "agnews":
        verbalizers = ["World", "Sports", "Business", "Tech"]  # num_classes
    elif dataset in ["sst-5"]:
        verbalizers = [
            "terrible",
            "bad",
            "okay",
            "good",
            "great",
        ]  
    elif dataset == "trec":
        verbalizers = [
            "Description",
            "Entity",
            "Expression",
            "Human",
            "Location",
            "Number",
        ]
    return verbalizers


