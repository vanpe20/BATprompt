# 🎯BATprompt
This is the official code implementation of papler [Robustness-aware Automatic Prompt Optimization](http://arxiv.org/abs/2412.18196)

## 📃Abstract
The performance of Large Language Models (LLMs) is based on the quality of the prompts and the semantic and structural integrity information of the input data. However, current prompt generation methods primarily focus on generating prompts for clean input data, often overlooking the impact of perturbed inputs on prompt performance. To address this limitation, we propose BATprompt (By Adversarial Training prompt), a novel method for prompt generation designed to withstand input perturbations (such as typos in the input). Inspired by adversarial training techniques, BATprompt demonstrates strong performance on a variety of perturbed tasks through a two-step process: adversarial perturbation and iterative optimization on unperturbed input via LLM. Unlike conventional adversarial attack methods, BATprompt avoids reliance on real gradients or model parameters. Instead, it leverages the advanced reasoning, language understanding and self reflection capabilities of LLMs to simulate gradients, guiding the generation of adversarial perturbations and optimizing prompt performance. In our experiments, we evaluate BATprompt on multiple datasets across both language understanding and generation tasks. The results indicate that BATprompt outperforms existing prompt generation methods, delivering superior robustness and performance under diverse perturbation scenarios.

## 🔜Quick Start

### ⚙️Settings
· **Evironment**: `pip install -r requirements.txt`
· **Data**: The data after adversarial perturbation is in the folder `./data`. To obtain the raw data, you can click here: [SAM](https://paperswithcode.com/dataset/samsum-corpus), [ASSET](https://github.com/facebookresearch/asset), [language-understanding](https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar)
· **OPENAI-API**: Please put the openai key in the file `./temp_file/key.yaml`

### 🏋️Adversarial Training
We provide adversarial training scripts under two perturbations, mix and combine, on the text summarization task.

```bash
bash script/train_sum_mix.sh 
bash script/train_sum_combine.sh 
```
Note that if you want to modify the task, please change the corresponding part of the script.


### 💭Inference

We provide test files for the text summarization task on the black-box model gpt and the white-box model llama under mix perturbation.

```bash
bash script/test_sum_gpt.sh # inference on gpt
bash script/test_sum_llama.sh # inference on llama
```
Note that if you want to switch the adversarial training method, change the `--attack_type` in the file.


### 📊Data Generate
We provide scripts for generating adversarial perturbation examples, which can be used to add a fixed type of perturbation to the text to generate a dataset.

```bash
bash script/create_data_sum.sh
```

### ❗️Note
While our method only supports black-box models such as gpt-3.5-turbo and gpt-4o-mini during training, white-box models such as llama2 can be used during testing.

### Parameters
When you execute the task, you may need to adjust the following parameters
· `datasets`: The datasets name used in the task.
· `task`: The task you want to generate instruction including **sum**, **sim**, **cls**.
· `iter`: The number of internal loops when the perturbation is added during the execution of each round of adversarial perturbation phase.
· `ad_iter`: Number of iteration rounds when generating instructions.
· `attack_type`: Type of perturbations
· `sample_num`: The number of samples used in each round during adversarial training.


### Code structure
```python
.
├── args.py
├── adversarial_attack.py # adversatial perturbation phase
├── data  # dataset, templates used
│   ├── cls
    │   ├── baseline.json #baseline prompt
    │   ├── prompt_ini.json # Initialize prompt
    │   └── template_cls.json # template in white-box model
│   ├── sim
│   └── sum  # wrapper
├── data_generate.py  # dataset generate
├── generate_prompt.py  # generate phase
├── llm_att.py  # llm query in first phase
├── llm_ge.py  # llm query in second phase
├── metrics.py  # metric calculation
├── process.py  # workflow of BATprompt
├── requirements.txt
├── run.py  # main file for BATprompt
├── scripts  # scripts to run the code
├── temp_file # template file in adversarial training
├── utils.py  # auxiliary functions
└── value.py #Inference process.
```

## 🙏 Citation

If you find this repository helpful, please consider citing our paper:

```
@misc{shi2024robustnessawareautomaticpromptoptimization,
      title={Robustness-aware Automatic Prompt Optimization}, 
      author={Zeru Shi and Zhenting Wang and Yongye Su and Weidi Luo and Fan Yang and Yongfeng Zhang},
      year={2024},
      eprint={2412.18196},
      archivePrefix={arXiv},
}
```

## Acknowledgements

This work is based on the following repos. Thanks for open-sourcing!

- [Evoprompt](https://github.com/beeevita/EvoPrompt)
- [APE](https://github.com/keirp/automatic_prompt_engineer)


