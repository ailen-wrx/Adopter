import os


res_file_0 = open('groundtruth.tsv', 'r')
res_file_1 = open('res_all_models.txt', 'r')
res_file_2 = open('res_comby_all.txt', 'r')
res_file_1_1 = open('res_ablation_1.txt', 'r')

res_file_3 = open('stat_comby.tsv', 'w')
res_file_4 = open('stat_adopter.tsv', 'w')
res_file_4_1 = open('stat_ablation_1.tsv', 'w')

gt = {}
exp = {}

def get_models():
    models = []
    for dir in os.listdir("../models/huggingface_transformers/src/transformers/models"):
        if os.path.exists(f'../models/huggingface_transformers/src/transformers/models/{dir}/modeling_{dir}.py'):
            models.append(f'../models/huggingface_transformers/src/transformers/models/{dir}/modeling_{dir}.py')
    return models


models = get_models()
models.sort()

for l in res_file_0.readlines():
    l = l.strip()
    if l.split('\t')[0] == 'model':
        continue
    rule_id = l.split('\t')[0]
    gt[rule_id] = l.split('\t')[1:]

res_file_0.close()


res_file_3.write('rule\tTP\tFN\tFP\n')
for l in res_file_2.readlines():
    l = l.strip()
    if l.split('\t')[0] == 'model':
        continue
    rule_id = l.split('\t')[0]
    for rule in gt.keys():
        if rule in rule_id:
            compare = gt[rule]
    baseline = l.split('\t')[1:]

    TP = 0
    FP = 0
    FN = 0
    res_file_3.write(f'{rule_id}\t')

    for i in range(len(compare)):
        c = int(compare[i])
        b = int(baseline[i])
        if c == b:
            TP += c
        elif c > b:
            TP += b
            FP += c - b
        elif c < b:
            TP += c
            FN += b - c

    res_file_3.write(f'{TP}\t{FP}\t{FN}\t')
    res_file_3.write('\n')

res_file_3.close()

res_file_4.write('model\tTP\tFN\tFP\n')
for l in res_file_1.readlines():
    l = l.strip()
    if l.split('\t')[0] == 'model':
        continue
    rule_id = l.split('\t')[0]
    for rule in gt.keys():
        if rule in rule_id:
            compare = gt[rule]
    baseline = l.split('\t')[1:]

    TP = 0
    FP = 0
    FN = 0
    res_file_4.write(f'{rule_id}\t')

    for i in range(len(compare)):
        c = int(compare[i])
        b = int(baseline[i])
        if c == b:
            TP += c
        elif c > b:
            TP += b
            FP += c - b
            print(f'FN: {rule_id} {i} {models[i]}')
        elif c < b:
            TP += c
            FN += b - c
            print(f'FP: {rule_id} {i} {models[i]}')

    res_file_4.write(f'{TP}\t{FP}\t{FN}\t')
    res_file_4.write('\n')

res_file_4.close()

res_file_4_1.write('model\tTP\tFN\tFP\n')
for l in res_file_1_1.readlines():
    l = l.strip()
    if l.split('\t')[0] == 'model':
        continue
    rule_id = l.split('\t')[0]
    for rule in gt.keys():
        if rule in rule_id:
            compare = gt[rule]
    baseline = l.split('\t')[1:]

    TP = 0
    FP = 0
    FN = 0
    res_file_4_1.write(f'{rule_id}\t')

    for i in range(len(compare)):
        c = int(compare[i])
        b = int(baseline[i])
        if c == b:
            TP += c
        elif c > b:
            TP += b
            FP += c - b
        elif c < b:
            TP += c
            FN += b - c

    res_file_4_1.write(f'{TP}\t{FP}\t{FN}\t')
    res_file_4_1.write('\n')

res_file_4_1.close()


res_file_1.close()
res_file_2.close()
res_file_1_1.close()