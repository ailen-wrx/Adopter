from IPython.display import clear_output
import os
import re
import json

import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from transformers import AutoConfig, PretrainedConfig
import torch


dump_fig_to_pdf = False
transformers = '/home/ubuntu/pytorch-opt/transformers-benchmarks/transformers'


def run_cmd(command, printoutput=True):
    import subprocess
    print("Running command:", command)
    completed_process = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    if not printoutput:
        if completed_process.stderr:
            print("Errors:", completed_process.stderr)
        else:
            print("Complete.")
        return
    print("Output:", completed_process.stdout)
    if completed_process.stderr:
        print("Errors:", completed_process.stderr)
    return


def apply_patch(patch):
    run_cmd('cd {transformers}; git stash push -- src/transformers/models/; git apply {patch}; git diff src/transformers/models/'.format(transformers=transformers, patch=patch))


def print_info():
    print('Pytorch version\t:', torch.__version__)
    print('PyTorch CUDA version\t:', torch.version.cuda)

    for i in range(torch.cuda.device_count()):
        print(f'GPU{i}\t\t:',torch.cuda.get_device_name(i))


def prepare_transformers():
    command = "cd {transformers}; git stash; git checkout 49d908aef2613af6f6a6569c6df420049ca48961".format(transformers=transformers)
    run_cmd(command, False)
    apply_patch("patches/trainer.patch")
    run_cmd('cd {transformers}; pip install -e ".[dev]" --no-deps 2>&1'.format(transformers=transformers), False)


@dataclass
class Exp:
    name: str           # Experiment name
    model: str          # huggingface model name
    batch_size: int     # batch size per GPU
    seq_len: int = None # input sequence length
    patch: str = None   # optional patch to apply before exp
        
    ## Improve speed / reduce memory  
    bf16: bool = False  # Faster, less memory. Recommend if GPU supports
    fp16: bool = False  # Faster, less memory, but need to scale loos. 
                        # Recommend if BF16 is not available.
    optim: str = 'adamw_hf'  # Optimization method
    grad_ckpt: bool = False  # save memory with an extra forward
    grad_accum: int = 1      # accumulate gradients for better performance
    steps: int = 20          # number of parameter updates
        
    ## Multi-GPUs
    gpus: str = '0'          # GPUs to use. "0,1" means use GPU 0 and 1
    tensor_para: int = 1     # Tensor parallelism
    deepspeed: bool = False  # if or not use deepspeed
    ds_config: str = ''      # deepspeed config
        
    ## kwargs
    kwargs: dict = None
        
    def __post_init__(self):         
        model_conf = AutoConfig.from_pretrained(self.model)
        get = lambda *keys: max([getattr(model_conf, k) if hasattr(model_conf, k) else 0 for k in keys])
        self.num_layers = get('num_hidden_layers', 'n_layer')
        self.num_gpus = len(self.gpus.split(','))                      
        self.hidden_size = get('hidden_size', 'n_embd', 'd_model')
        self.vocab_size = get('vocab_size')
        self.num_heads = get('num_attention_heads', 'n_head')
        if self.seq_len is None:
            self.seq_len = get('max_position_embeddings', 'n_ctx')
        n, h, s, v = self.num_layers, self.hidden_size, self.seq_len, self.vocab_size
        att, ffn, embed = 4*h*s**2 + 8*s*h**2, 16*s*h**2, 2*s*h*v
        forward = n*(att+ffn) + embed
        # TFLOPs to train one example
        self.tflops = (4 * forward if self.grad_ckpt else 3 * forward) / 1e12
        if self.deepspeed:            
            self.launcher = 'deepspeed'            
        else:
            self.launcher = f'torchrun --nproc_per_node {self.num_gpus}' 
            
    def print_results(self):
        print('Total samples / second\t: %.1f' % self.samples_per_sec)
        print('Per GPU memory (GB)\t: %.1f'% self.gpu_mem)
        print('Per GPU TFLOPs\t\t: %.1f' % (self.samples_per_sec * self.tflops / self.num_gpus))


def compare(exps, save_to=""):
    fig, ax = plt.subplots(ncols=3, figsize=(9,len(exps)/2))
    x = list(range(len(exps)))
    for i, (y, l) in enumerate((
        ([e.samples_per_sec for e in exps], 'Samples / sec'), 
        ([e.samples_per_sec * e.tflops / e.num_gpus for e in exps], 'per GPU TFLOPS'),
        ([e.gpu_mem for e in exps], 'per GPU memory (GB)'))):
        print(f"{l}: {['%.2f' % e for e in y]}")
        ax[i].barh(x, y, align='center', height=0.6, color=plt.get_cmap('Set1')(x))
        ax[i].invert_yaxis()
        ax[i].set_xlabel(l)
        if i == 0:
            ax[i].set_yticks(x, labels=[e.name for e in exps])
        else:
            ax[i].set_yticklabels([])

    if save_to:
        plt.savefig(save_to, bbox_inches="tight")
    else:
        plt.show()


def hf_bert(exp):
    cmd = f'''export CUDA_VISIBLE_DEVICES={exp.gpus}; \
{exp.launcher} transformers/examples/pytorch/language-modeling/run_mlm.py \
--config_name {exp.model} --tokenizer_name {exp.model} \
--dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
--do_train --max_seq_length {exp.seq_len} \
--per_device_train_batch_size {exp.batch_size} \
--fp16 {exp.fp16} --bf16 {exp.bf16} \
--optim {exp.optim} --max_steps {exp.steps} \
--gradient_accumulation_steps {exp.grad_accum} \
--gradient_checkpointing {exp.grad_ckpt} \
--output_dir /tmp/bert/ --overwrite_output_dir yes --skip_memory_metrics False'''
    return cmd


def hf_resnet(exp):
    cmd = f'''export CUDA_VISIBLE_DEVICES={exp.gpus}; \
{exp.launcher} transformers/examples/pytorch/image-classification/run_image_classification.py \
--model_name_or_path {exp.model} \
--dataset_name beans \
--do_train \
--per_device_train_batch_size {exp.batch_size} \
--fp16 {exp.fp16} --bf16 {exp.bf16} \
--optim {exp.optim} --max_steps {exp.steps} \
--gradient_accumulation_steps {exp.grad_accum} \
--gradient_checkpointing {exp.grad_ckpt} \
--output_dir /tmp/beans/ --overwrite_output_dir yes --skip_memory_metrics False \
--remove_unused_columns False \
--learning_rate 2e-5 \
--ignore_mismatched_sizes'''
    return cmd


def hf_groupvit(exp):
    cmd = f'''export CUDA_VISIBLE_DEVICES={exp.gpus}; \
{exp.launcher} transformers/examples/pytorch/contrastive-image-text/run_clip.py \
--model_name_or_path {exp.model} \
--dataset_name ydshieh/coco_dataset_script \
--dataset_config_name=2017 --data_dir dummy_data \
--image_column image_path --caption_column caption \
--do_train \
--per_device_train_batch_size={exp.batch_size} \
--fp16 {exp.fp16} --bf16 {exp.bf16} \
--optim {exp.optim} --max_steps {exp.steps} \
--gradient_accumulation_steps {exp.grad_accum} \
--gradient_checkpointing {exp.grad_ckpt} \
--output_dir ./groupvit \
--remove_unused_columns=False \
--learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
--overwrite_output_dir --skip_memory_metrics False'''
    return cmd


def hf_gptneo(exp):
    cmd = f'''export CUDA_VISIBLE_DEVICES={exp.gpus}; \
{exp.launcher} transformers/examples/pytorch/language-modeling/run_clm.py \
--config_name {exp.model} --tokenizer_name {exp.model} \
--dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
--do_train \
--per_device_train_batch_size {exp.batch_size} \
--fp16 {exp.fp16} --bf16 {exp.bf16} \
--optim {exp.optim} --max_steps {exp.steps} \
--gradient_accumulation_steps {exp.grad_accum} \
--gradient_checkpointing {exp.grad_ckpt} \
--output_dir /tmp/gptneo/ --overwrite_output_dir yes --skip_memory_metrics False'''
    return cmd


def hf_t5(exp):
#  cmd = f'''export CUDA_VISIBLE_DEVICES={exp.gpus}; \
# {exp.launcher} transformers/examples/pytorch/language-modeling/run_clm.py \
# --config_name {exp.model} --tokenizer_name {exp.model} \
# --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
# --do_train \
# --per_device_train_batch_size {exp.batch_size} \
# --fp16 {exp.fp16} --bf16 {exp.bf16} \
# --optim {exp.optim} --max_steps {exp.steps} \
# --gradient_accumulation_steps {exp.grad_accum} \
# --gradient_checkpointing {exp.grad_ckpt} \
# --output_dir /tmp/gptneo/ --overwrite_output_dir yes --skip_memory_metrics False'''
    
    cmd = f"""export CUDA_VISIBLE_DEVICES={exp.gpus}; \
    {exp.launcher} transformers/examples/pytorch/translation/run_translation.py \
    --model_name_or_path {exp.model} \
    --do_train \
    --source_lang en \
    --target_lang ro \
    --source_prefix "translate English to Romanian: " \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --output_dir /tmp/tst-translation \
    --per_device_train_batch_size={exp.batch_size} \
    --overwrite_output_dir \
    --predict_with_generate \
    --fp16 {exp.fp16} --bf16 {exp.bf16} \
    --optim {exp.optim} --max_steps {exp.steps} \
    --gradient_accumulation_steps {exp.grad_accum} \
    --gradient_checkpointing {exp.grad_ckpt} \
    --output_dir /tmp/t5/ --overwrite_output_dir yes --skip_memory_metrics False"""
    return cmd


model_to_hf = {'bert': hf_bert, 'resnet': hf_resnet, 'groupvit': hf_groupvit,
                'gpt': hf_gptneo, 't5': hf_t5}

def hf(exp):
    # prepare_transformers()
    if exp.patch is not None:
        apply_patch(exp.patch)
    for (model, fun) in model_to_hf.items():
        if exp.model.__contains__(model):
            cmd = fun(exp)
            if exp.deepspeed:
                cmd += f' --deepspeed {exp.ds_config}'
            if exp.kwargs is not None and "flags" in exp.kwargs:
                cmd += " " + " ".join(exp.kwargs["flags"])
            cmd += ' > log/log_{name}.txt 2>&1'.format(name=exp.name)
            run_cmd(cmd)
            ret = hf_log(exp, 'log/log_{name}.txt'.format(name=exp.name))
            if ret is not None:
                ret.print_results()
            return ret
            
    
def hf_log(exp, log_filename):
    with open(log_filename) as f:
        lines = f.readlines()
        
    global_batch_size = 0
    for l in lines:
        if 'CUDA out of memory' in l:
            print('Out of GPU memory, try a smaller batch size')
            return None
        if 'Total train batch size' in l:
            global_batch_size = int(next(iter(reversed(re.findall('= +([\d\.]+)', l))), 0))
        if '{\'train_runtime' in l:
            if global_batch_size == 0:
                print(f'Failed to parse global batch size. Check {log_filename} to find error')
            metrics = json.loads(l.replace('\'', '\"'))
            exp.gpu_mem = (metrics['init_mem_cpu_peaked_delta'] + \
                    metrics['train_mem_gpu_alloc_delta'] + metrics['train_mem_gpu_peaked_delta']) / 1e9
            if 'step_time_list' in metrics:
                step_time_list = metrics['step_time_list']
                # Remove the first 5 iterations (warmup)
                # step_time_list = step_time_list[5:] if len(step_time_list) > 5 else step_time_list
                exp.samples_per_sec = (global_batch_size * len(step_time_list)) / sum(step_time_list)
            else:
                print("Cannot find 'step_time_list', use HF Triner reported samples/sec")
                exp.samples_per_sec = metrics['train_samples_per_second']
            return exp
    print(f'Failed. Check "{log_filename}" to find error')    
    return None
