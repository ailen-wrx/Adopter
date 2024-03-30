import os


PATH='huggingface_transformers/src/transformers/models'
SRC='hf_models'

exists = 0
cnt = 0
for dir in os.listdir(PATH):
    src_path=os.path.join(PATH,dir,f'modeling_{dir}.py')
    if os.path.exists(src_path):
        os.system(f'cp {src_path} {SRC}/{dir}.py')
        exists += 1
    cnt += 1

print(f'exists: {exists}/{cnt}')

