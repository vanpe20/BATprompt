from sacrebleu.metrics import BLEU, CHRF, TER
from rouge import Rouge
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from easse.sari import corpus_sari
from mosestokenizer import *
import Levenshtein

bleu = BLEU(tokenize='zh')
def cal_bleu(bleu_model,output_texts, ref_texts):
    bleu_score = bleu_model.corpus_score(output_texts, ref_texts).score
    return bleu_score

def cal_cls_score(pred_list, label_list,metric='acc'):
    pred_list = [p.lower() for p in pred_list]
    label_list = [l.lower() for l in label_list]
    if metric == 'f1':
        score = f1_score(label_list, pred_list, average='macro')
    elif metric == 'acc':
        score = accuracy_score(label_list, pred_list)
    return score

def cal_rouge(output_texts, ref_texts):

    rouge = Rouge()

    if isinstance(output_texts, list):
        output = output_texts
        ref = ref_texts
    else:
        output = []
        ref = []
        output.append(output_texts)
        ref.append(ref_texts)

    scores = rouge.get_scores(output, ref, avg=True)

    return scores['rouge-1']['f'], scores['rouge-2']['f'], scores['rouge-l']['f'] 

def cal_sari(orig_sents, sys_sents, refs_sents, makeadv = True):

    if not isinstance(orig_sents, list):
        orig_sents = [orig_sents]
    if not isinstance(sys_sents, list):
        sys_sents = [sys_sents]
    if makeadv:
        refs_sents = [[ref] for ref in refs_sents] 
    sari = corpus_sari(orig_sents=orig_sents,  
                sys_sents=sys_sents, 
                refs_sents=refs_sents)

    return sari

def calculate_similarity(data1, data2):
    distance = Levenshtein.distance(data1, data2)
    max_length = max(len(data1), len(data2))
    similarity = (1 - distance / max_length) 
    return similarity
