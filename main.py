import copy, os, sys
import time as time_tool
from dsl import refactorHelper
from patcher import *


RULE1 = [
    'BertSelfAttention($hs, $am, $hm, $ehs, $eam, $pkv, $oa) -> $o => _',
    '@BertSelfAttention($c, position_embedding_type=$pet) -> $s => @epoi.ops.xformers_attn.BertSelfAttentionWithXF($c, position_embedding_type=$pet, attn_op_name="cutlass") -> $s'
]
RULE2 = [
    'T5Attention($hs, mask=$am, position_bias=$pb, layer_head_mask=$lhm, past_key_value=$pkv, use_cache=$c, output_attentions=$oa) -> o => _',
    '@T5Attention($c, has_relative_attention_bias=$hrab) -> $s => @epoi.ops.xformers_attn.T5Attention($c, has_relative_attention_bias = $hrab, attn_op_name="cutlass") -> $s'
]
RULE3 = [
    'GPT2Attention($hs, layer_past=$lp, attention_mask=$am, head_mask=$hm, use_cache=$c, output_attentions=$oa) -> o => _',
    '@GPT2Attention($c, is_cross_attention=$cr, layer_idx=$li) -> $s => @epoi.ops.xformers_attn.GPT2AttentionWithXF($c, is_cross_attention=$cr, layer_idx=$li) -> $s'
]
RULE4 = [
    'torch.nn.functional.softmax($as, dim=-1) -> $ap => xformers.triton.softmax.softmax($as) -> $ap'
]
RULE5 = [
    'torch.nn.Dropout($hs) -> $hs; $hs + $it -> $h; torch.nn.LayerNorm($h) -> $hs => epoi.ops.torchscript_ops.FusedDropoutAddLayerNorm($hs, $it) -> $hs',
    '@torch.nn.Dropout($chdp) -> _; @torch.nn.LayerNorm($chs, eps=$clne) -> _ => @epoi.ops.torchscript_ops.FusedDropoutAddLayerNorm($chs, $chdp, eps=$clne) -> self.dropout_add_layernorm'
]
RULE6 = [
    'torch.nn.Linear($hs) -> $hs; ACT2FN[config.hidden_act]($hs) -> $hs => _',
    '@torch.nn.Linear($chs, $cis, bias=True) -> $sd; @ACT2FN[config.hidden_act] -> $siaf => @torch.nn.Linear($chs, $cis, bias=False) -> $sd; @epoi.ops.torchscript_ops.FusedBiasGELU($cis, prev_weight=$sd.weight) -> $siaf'
]
RULE7 = [
    'torch.nn.Conv2d($x) -> $h; torch.nn.BatchNorm2d($h) -> $h => epoi.ops.torchscript_ops.FusedConv2dBatchNorm2d($x) -> $h',
    '@torch.nn.Conv2d(in_channels=$ic, out_channels=$oc, kernel_size=$ks, stride=$st, bias=$bs) -> _; @torch.nn.BatchNorm2d(num_features=$oc) -> _ => @epoi.ops.torchscript_ops.FusedConv2dBatchNorm2d($ic, $oc, $ks, stride=$st, bias=$bs) -> self.conv_batchnorm'
]
RULE8 = [
    'torch.nn.Linear($hs) -> $hs; torch.nn.BatchNorm1d($hs) -> $hs => torch.nn.intrinsic.qat.modules.LinearBn1d($hs) -> $hs',
    '@torch.nn.Linear($in, $out, $b) -> _; @torch.nn.BatchNorm1d($out) -> _ => @torch.nn.intrinsic.qat.modules.LinearBn1d($in, $out, $b) -> self.linear_bn'
]
RULE9 = [
    'torch.nn.Linear($hs) -> $q; torch.nn.Linear($hs) -> $k; torch.nn.Linear($hs) -> $v; self._split_heads($q, $nh, $hd) -> $q; self._split_heads($k, $nh, $hd) -> $k; self._split_heads($v, $nh, $hd) -> $v => slapo.op.linear.FusedQKV($hs) -> $q, $k, $v',
    '@torch.nn.Linear($sed, $sed, bias=False) -> _; @torch.nn.Linear($sed, $sed, bias=False) -> _; @torch.nn.Linear($sed, $sed, bias=False) -> _ => @slapo.op.linear.FusedQKV($sed, self.num_heads, 1) -> self.qkv'
]


MODULE_0 = 'models.huggingface_transformers.src.transformers.models.bert.model'

MODULE_1 = 'models.huggingface_transformers.src.transformers.models.bert.modeling_bert'
MODULE_2 = 'models.huggingface_transformers.src.transformers.models.roberta.modeling_roberta'
MODULE_3 = 'models.huggingface_transformers.src.transformers.models.gpt2.modeling_gpt2'
MODULE_4 = 'models.huggingface_transformers.src.transformers.models.gpt_neo.modeling_gpt_neo'
MODULE_5 = 'models.huggingface_transformers.src.transformers.models.distilbert.modeling_distilbert'
MODULE_6 = 'models.huggingface_transformers.src.transformers.models.t5.modeling_t5'
MODULE_7 = 'models.huggingface_transformers.src.transformers.models.albert.modeling_albert'
MODULE_8 = 'models.huggingface_transformers.src.transformers.models.groupvit.modeling_groupvit'
MODULE_9 = 'models.huggingface_transformers.src.transformers.models.resnet.modeling_resnet'


def get_models():
    models = []
    for dir in os.listdir("models/huggingface_transformers/src/transformers/models"):
        if not os.path.exists(f'models/huggingface_transformers/src/transformers/models/{dir}/modeling_{dir}.py'):
            continue
        models.append(f'models.huggingface_transformers.src.transformers.models.{dir}.modeling_{dir}')
    return models

def main():
    args = sys.argv[1:]
    ablation = None
    working_dir = os.getcwd()

    LOG_FILE = ''
    RES_FILE = ''
    TIME_FILE = ''
    if len(args) > 0 and args[0] == 'all':
        LOG_FILE = 'results/log_all_models.txt'
        RES_FILE = 'results/res_all_models.txt'
        TIME_FILE = 'results/time.tsv'
        models = get_models()
        models.sort()
        modules = models
        print(len(models))
    else:
        LOG_FILE = 'results/log_10_models.txt'
        RES_FILE = 'results/res_10_models.txt'
        TIME_FILE = 'results/time_10_models.tsv'
        modules = [MODULE_1, MODULE_2, MODULE_3, MODULE_4, MODULE_5, MODULE_6, MODULE_7, MODULE_8, MODULE_9]

    if len(args) > 1 and args[1] == 'ablation_1':
        ablation = 'path'
        LOG_FILE = 'results/log_ablation_1.txt'
        RES_FILE = 'results/res_ablation_1.txt'
        TIME_FILE = 'results/time_ablation_1.tsv'
    elif len(args) > 1 and args[1] == 'ablation_2':
        ablation = 'inter'
        LOG_FILE = 'results/log_ablation_2.txt'
        RES_FILE = 'results/res_ablation_2.txt'
        TIME_FILE = 'results/time_ablation_2.tsv'
    
    log_file_1 = open(LOG_FILE, 'w')
    res_file_1 = open(RES_FILE, 'w')
    time_file_1 = open(TIME_FILE, 'w')
    All_Rules = [RULE1, RULE2, RULE3, RULE4, RULE5, RULE6, RULE7, RULE8, RULE9]

    # All_Rules = [RULE6]
    # modules = [MODULE_1]
    # log_file_1 = open('results/log.txt', 'w')
    # res_file_1 = open('results/res.txt', 'w')

    results = []
    times = []

    if os.path.exists(f'{working_dir}/{RESULT_DIR}'):
        run_cmd(f'rm -r {working_dir}/{RESULT_DIR}')
    if os.path.exists(f'{working_dir}/{PATCH_DIR}'):
        run_cmd(f'rm -r {working_dir}/{PATCH_DIR}')

    for ri, Rules in enumerate(All_Rules):
        ri += 1
        R = refactorHelper(Rules)
        for rule in Rules:
            log_file_1.write(f"[rule_{ri}] {rule}\n")
            print(f"[rule_{ri}] {rule}")

        result = []
        time = []
        for mi, module in enumerate(modules):
            mi += 1
            log_file_1.write(f"[rule_{ri}][module_{mi}] {module}\n")
            print(f"[rule_{ri}][module_{mi}] {module}")
            time0 = time_tool.perf_counter()
            try:
                S = ModelVisitor(module, ablation=ablation)
            except:
                continue
            S.begin_visit()
            patches = []

            for cls_idx in S.call_sequences:
                # for rule in R.rules[:1]:
                rule = R.rules[0]
                _call_sequence = []
                for func in S.call_sequences[cls_idx]:
                    _call_sequence += S.call_sequences[cls_idx][func]['_call_seqs']
                matches = set()
                s_matches = set()
                for call_sequence in _call_sequence:
                    locations = rule.patternMatch(call_sequence.call_statements)
                    if len(locations) > 0:
                        locations = locations
                        for idx, (ii, jj, arg_dict) in enumerate(locations):
                            if ii != -1:
                                matched_call_sequence = call_sequence.call_statements[ii: jj]
                                s_affected_lines = []
                                s_refactored_code = []
                                if set([call_stmt.in_seq for call_stmt in matched_call_sequence]) != {True} and len(R.rules) > 1:
                                    alias_map = {}
                                    for matched_call_stmt in matched_call_sequence:
                                        if matched_call_stmt.refactoredFunctionId and matched_call_stmt.functionId:
                                            for key in matched_call_stmt.refactoredFunctionId:
                                                if not key in alias_map:
                                                    alias_map[key] = set()
                                                alias_map[key].add('.'.join(matched_call_stmt.functionId))
                                    rule_1 = R.rules[1]
                                    for i_call_sequence in _call_sequence:
                                        match, arg_dict_1 = rule_1.unorderedPatternMatch(i_call_sequence.call_statements, alias_map, ablation=ablation)
                                        if match:
                                            # arg_dict.update(arg_dict_1)
                                            affected_lines_1, refactored_code_1 = rule_1.unorderedRefactor(i_call_sequence, match, arg_dict_1)
                                            s_lineno_str = ','.join([str(l) for l in affected_lines_1])
                                            if s_lineno_str in s_matches:
                                                continue
                                            s_matches.add(s_lineno_str)
                                            if affected_lines_1 and refactored_code_1:
                                                s_affected_lines.append(affected_lines_1)
                                                s_refactored_code.append(refactored_code_1)
                                affected_lines, refactored_code = rule.refactor(call_sequence, ii, jj, arg_dict)
                                if affected_lines and refactored_code:
                                    s_affected_lines.append(affected_lines)
                                    s_refactored_code.append(refactored_code)
                                    lineno_str = ','.join([str(l) for l in affected_lines])
                                    if lineno_str in matches:
                                        continue
                                    matches.add(lineno_str)

                                if len(s_affected_lines) > 0:
                                    patch = Patch(module)
                                    for c_idx in range(len(s_affected_lines)):
                                        patch.add_change(Change(s_affected_lines[c_idx], s_refactored_code[c_idx]))
                                    patches.append(patch)
                                    s_affected_lines.clear()
                                    s_refactored_code.clear()

            for pid, patch in enumerate(patches):
                patch.write_patch(ri, pid+1)

            time1 = time_tool.perf_counter()
            
            _time = time1 - time0

            log_file_1.write(f"[rule_{ri}][module_{mi}][stat] {len(patches)}\n")
            print(f"[rule_{ri}][module_{mi}][stat] {len(patches)}\n")
            print(f"[rule_{ri}][module_{mi}][time] {_time:.2f} seconds")
            result.append(len(patches))
            time.append(_time)
        results.append(result)
        times.append(time)
        log_file_1.write(f"\n")
        print('')

    res_file_1.write('model\t')
    for i, module in enumerate(modules):
        res_file_1.write(f'{i+1}\t')
    res_file_1.write('\n')
    for i, rule in enumerate(All_Rules):
        res_file_1.write(f'rule_{i+1}\t')
        for j, module in enumerate(modules):
            res_file_1.write(f'{results[i][j]}\t')
        res_file_1.write('\n')

    time_file_1.write('model\t')
    for i, module in enumerate(modules):
        time_file_1.write(f'{i+1}\t')
    time_file_1.write('\n')
    for i, rule in enumerate(All_Rules):
        time_file_1.write(f'rule_{i+1}\t')
        for j, module in enumerate(modules):
            time_file_1.write(f'{times[i][j]}\t')
        time_file_1.write('\n')


if __name__ == '__main__':
    main()
