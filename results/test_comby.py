import subprocess
import os

PATCH_DIR = 'patches_comby_all'


def system_call(command):
    p = subprocess.Popen([command], stdout=subprocess.PIPE, shell=True)
    return p.stdout.read()

model1 = '../models/huggingface_transformers/src/transformers/models/bert/modeling_bert.py'
model2 = '../models/huggingface_transformers/src/transformers/models/roberta/modeling_roberta.py'
model3 = '../models/huggingface_transformers/src/transformers/models/gpt2/modeling_gpt2.py'
model4 = '../models/huggingface_transformers/src/transformers/models/gpt_neo/modeling_gpt_neo.py'
model5 = '../models/huggingface_transformers/src/transformers/models/distilbert/modeling_distilbert.py'
model6 = '../models/huggingface_transformers/src/transformers/models/t5/modeling_t5.py'
model7 = '../models/huggingface_transformers/src/transformers/models/albert/modeling_albert.py'
model8 = '../models/huggingface_transformers/src/transformers/models/groupvit/modeling_groupvit.py'
model9 = '../models/huggingface_transformers/src/transformers/models/resnet/modeling_resnet.py'
models = [model1, model2, model3, model4, model5, model6, model7, model8, model9]

rule1 = [':[[v1]].:[[v2]] = BertSelfAttention(:[[v3]], position_embedding_type=:[[v4]])', 'from epoi.ops.xformers_attn import BertSelfAttention as xformers_attn_BertSelfAttention; :[[v1]].:[[v2]] = xformers_attn_BertSelfAttention(:[[v3]], position_embedding_type=:[[v4]], attn_op_name="cutlass")']
rule2 = [':[[v1]].:[[v2]] = T5Attention(:[[v3]], has_relative_attention_bias=:[[v4]])', 'from epoi.ops.xformers_attn import T5Attention as xformers_attn_T5Attention; :[[v1]].:[[v2]] = xformers_attn_T5Attention(:[[v3]], has_relative_attention_bias=:[[v4]], attn_op_name="cutlass")']
rule3 = [':[[v1]].:[[v2]] = GPT2Attention(:[[v3]], is_cross_attention=:[[v4]], layer_idx=:[[v5]])', 'from epoi.ops.xformers_attn import GPT2Attention as xformers_attn_GPT2Attention; :[[v1]].:[[v2]] = xformers_attn_T5Attention(:[[v3]], is_cross_attention=:[[v4]], layer_idx=:[[v5]], attn_op_name="cutlass")']
rule4 = [':[[v1]] = nn.functional.softmax(:[[v3]], dim=-1)', 'from xformers.triton.softmax import softmax as softmax_softmax; :[[v1]] = softmax_softmax(:[[v3]])']
rule5_1 = [':[[v1]] = self.dropout(:[[v1]]) :[[v1]] = self.LayerNorm(:[[v1]] + :[[v2]])', 'self.fused_dropout_layernorm(:[[v1]] + :[[v2]])']
rule5_2 = [':[[v1]].:[[v2]] = nn.LayerNorm(:[[v3]].:[[v4]], eps=:[[v3]].:[[v5]]) :[[v1]].:[[v6]] = nn.Dropout(:[[v3]].:[[v7]])', 'from epoi.ops.torchscript_ops import FusedDropoutAddLayerNorm; :[[v1]].fused_dropout_layernorm = epoi.ops.torchscript_ops.FusedDropoutAddLayerNorm(:[[v3]].:[[v4]], :[[v3]].:[[v7]], :[[v3]].:[[v5]])']
rule6 = [':[[v1]].:[[v2]] = nn.Linear(config.hidden_size, config.intermediate_size) if isinstance(config.hidden_act, str): self.intermediate_act_fn = ACT2FN[config.hidden_act] else: self.intermediate_act_fn = config.hidden_act',\
         ':[[v1]].:[[v2]] = nn.Linear(config.hidden_size, config.intermediate_size, bias=False); assert (config.hidden_act == \'gelu\'); self.intermediate_act_fn = torchscript_ops_FusedBiasGELU(config.intermediate_size, prev_weight=:[[v1]].:[[v2]].weight)']
rule7 = [':[[v1]].:[[v2]] = nn.Conv2d(:[[v3]], :[[v4]], kernel_size=:[[v5]], stride=:[[v6]], bias=:[[v7]]) :[[v1]].:[[v8]] = nn.BatchNorm2d(:[[v4]])', 'self.conv_batchnorm = torchscript_ops_FusedConv2dBatchNorm2d(:[[v3]], :[[v4]], :[[v5]], stride=:[[v6]], bias=:[[v7]])']
rule8 = [':[[v1]].:[[v2]] = nn.Sequential(nn.Linear(self.vision_embed_dim, self.projection_intermediate_dim, bias=True),nn.BatchNorm1d(self.projection_intermediate_dim),nn.ReLU(inplace=True),nn.Linear(self.projection_intermediate_dim, self.projection_dim, bias=True))', 'from torch.nn.intrinsic.qat.modules import LinearBn1d as modules_LinearBn1d; :[[v1]].:[[v2]] = nn.Sequential(modules_LinearBn1d(self.vision_embed_dim, self.projection_intermediate_dim),nn.ReLU(inplace=True),nn.Linear(self.projection_intermediate_dim, self.projection_dim, bias=True))']
rule9_1 = [':[[v2]] = self.q_proj(:[[v1]]) :[[v3]] = self.k_proj(:[[v1]]) :[[v4]] = self.v_proj(:[[v1]])', '(:[[v2]], :[[v3]], :[[v4]]) = self.qkv(:[[v1]])']
rule9_2 = ['self.:[[v1]] = nn.Linear(self.:[[v2]], self.:[[v2]], bias=:[[v3]]) self.:[[v4]] = nn.Linear(self.:[[v2]], self.:[[v2]], bias=:[[v3]]) self.:[[v5]] = nn.Linear(self.:[[v2]], self.:[[v2]], bias=:[[v3]])', 'from slapo.op.linear import FusedQKV as linear_FusedQKV; self.qkv = linear_FusedQKV(self.:[[v2]], self.num_heads, 1)']

rule_id = ['1', '2', '3', '4', '5_1', '5_2', '6', '7', '8', '9_1', '9_2']
rules = [rule1, rule2, rule3, rule4, rule5_1, rule5_2, rule6, rule7, rule8, rule9_1, rule9_2]
# rules = [rule9_1]

def get_models():
    models = []
    for dir in os.listdir("../models/huggingface_transformers/src/transformers/models"):
        if os.path.exists(f'../models/huggingface_transformers/src/transformers/models/{dir}/modeling_{dir}.py'):
            models.append(f'../models/huggingface_transformers/src/transformers/models/{dir}/modeling_{dir}.py')
    return models


models = get_models()
models.sort()
# models = [f'../models/huggingface_transformers/src/transformers/models/gpt_neo/modeling_gpt_neo.py']
print(len(models))

res_file_1 = open('res_comby_all.txt', 'w')
log_file_1 = open('log_comby_all.txt', 'w')

if not os.path.exists(PATCH_DIR):
    os.mkdir(PATCH_DIR)

results = []
for i, rule in enumerate(rules):
    result = []
    log_file_1.write('[RULE ' + rule_id[i] + ']\n')
    print('[RULE ' + rule_id[i] + ']')
    for j, model in enumerate(models):
        model_id = model.split('modeling_')[-1].split('.py')[0]
        log_file_1.write('[RULE ' + rule_id[i] + '][MODEL ' + str(j+1) + ']\n')
        print('[RULE ' + rule_id[i] + '][MODEL ' + str(j+1) + '] ' + model_id)
        print(f'comby -diff \'{rule[0]}\' \'{rule[1]}\' {model}')
        out = system_call(f'comby -diff \'{rule[0]}\' \'{rule[1]}\' {model}').decode('ascii')
        print(out)
        if out.strip() != '':
            with open(f'{PATCH_DIR}/{rule_id[i]}_{model_id}.patch', 'w') as f:
                f.write(out)
        log_file_1.write(out)
        result.append(out.count('@@ -'))
    results.append(result)

res_file_1.write('model\t')
for i, module in enumerate(models):
    res_file_1.write(f'{i+1}\t')
res_file_1.write('\n')
for i, rule in enumerate(rules):
    res_file_1.write(f'rule_{rule_id[i]}\t')
    for j, module in enumerate(models):
        res_file_1.write(f'{results[i][j]}\t')
    res_file_1.write('\n')