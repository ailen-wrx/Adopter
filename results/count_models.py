import os


def get_models():
    models = []
    for dir in os.listdir("../models/huggingface_transformers/src/transformers/models"):
        if os.path.exists(f'../models/huggingface_transformers/src/transformers/models/{dir}/modeling_{dir}.py'):
            models.append(f'../models/huggingface_transformers/src/transformers/models/{dir}/modeling_{dir}.py')
    return models


models = get_models()
models.sort()
model_id = [model.split('modeling_')[-1].split('.py')[0] for model in models]

print('model\t', end='')
for i, module in enumerate(models):
    print(f'{model_id[i]}\t', end='')
print()