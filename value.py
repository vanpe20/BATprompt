from args import parse_args
from utils import set_seed
from adversarial_attack import Opentask
from generate_prompt import Opentask_Ge
from process import Adversarial, Open_optimize
import logging
import json


def run(args,prompt=None):
    set_seed(args.seed)
    attack =  Opentask(args)
    generate_task = Opentask_Ge(args)

    generate = generate_task
    gen = Open_optimize(args,generate=generate)
    idx = []

    scores = []

    

    prompt = args.prompt
    prompt = [prompt]
    weak_file = "./temp_file/weak.json"
    weak_type = json.load(open(weak_file, "r"))
    weak = weak_type[args.task][args.attack_type]

    if args.language_model == 'gpt':

        for j, cls in enumerate(weak):
            score1 = gen.black_evaluate(prompt, attack_class= cls, judge=True)
            scores.append(score1)
    
        return scores

    
    else:
        for j, cls in enumerate(weak):
            score1 = gen.white_evaluate(prompt, attack_class= cls, judge=True)
            scores.append(score1)
        return scores



if __name__ == "__main__":

    args = parse_args()
    scores = run(args)

