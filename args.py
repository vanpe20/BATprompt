import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="training args.")
    # prompt args
    parser.add_argument("--prompt", type=str, default = None, help="The initial prompt of task")
    parser.add_argument("--dataset", type=str, default="sam", help="dataset name")
    parser.add_argument("--task", type=str, choices=["cls", "sum", "sim"])
    parser.add_argument("--test_file", type=str, default=None)
    parser.add_argument("--generate", type=int, default=2)

    parser.add_argument(
        "--language_model",
        type=str,
        help="model for task test, e.g., llama2, gpt",
    )
    parser.add_argument("--seed", type=int, default=5, help="random seed")
    parser.add_argument("--llm_type", type=str, default="turbo", help='llm to generate prompt', choices=['turbo', 'gpt4'])
    parser.add_argument(
        "--sample_num",
        type=int,
        default=5,
        help="number of samples used to do the adversarial training",
    )
    parser.add_argument("--ad_iter", type=int, default="adversarial training iteration times")
    parser.add_argument("--iter", type=int, default="attack iteration times")
    parser.add_argument("--data_gen", type=bool, default=False)
    parser.add_argument("--setting", type=str, default="default", help="setting of the OpenAI API")
    parser.add_argument('--attack_type',type=str, choices=["mix", "combine"])
    args = parser.parse_args()
    return args



