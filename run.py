from args import parse_args
from utils import set_seed
from adversarial_attack import Opentask
from generate_prompt import Opentask_Ge
from process import Adversarial, Open_optimize
import logging
import json


def train(args):
    set_seed(args.seed)
    advattack = Opentask(args)

    generate_task = Opentask_Ge(args)

    attack = advattack
    adv = Adversarial(args, attack=attack)
    generate = generate_task
    gen = Open_optimize(args,generate=generate)
    weak_file = "./temp_file/weak.json"
    weak_type = json.load(open(weak_file, "r"))
    weak = weak_type[args.task][args.attack_type]
    instructions = []
    instruction = None

    for i in range(args.ad_iter):
        idx, answer= adv.forward(datagen=False, weak = weak, instruction=instruction)

        results, token= gen.forward(idx, weak, answer, instruction)

        if i < args.ad_iter - 1:
            instruction = results[0]
            instructions.append(instruction)

    instructions.extend(results)

    return instructions

    

if __name__ == "__main__":


    args = parse_args()
    train(args)





