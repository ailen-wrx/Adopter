from utils import *
from pattern_matcher import *

RESULT_DIR = 'results/optimized_models'
PATCH_DIR = 'results/patches'


class Change(object):
    def __init__(self, lines, codes):
        self.affected_lines = lines
        self.codes = codes


class Patch(object):
    def __init__(self, module_name):
        self.working_dir = os.getcwd()
        self.module_name = module_name
        self.module_id = module_name.split('.modeling_')[-1]
        self.model_path = module_name.replace('.', '/') + '.py'
        self.model_dir = '/'.join(self.model_path.split('/')[:-1])
        self.changes = []
        self.affected_lines = []
        self.change_loc = {}
        self.align = {}

    def add_change(self, change):
        self.changes.append(change)
        self.affected_lines += change.affected_lines
        self.change_loc[change.affected_lines[0]] = change

    def print(self):
        print(f'Patch for {self.model_path}')
        for change in self.changes:
            print(f'Affected lines: {change.affected_lines}')
            print(f'Code: {[astunparse.unparse(c).strip() for c in change.codes]}')
            print('')

    def write_patch(self, rid, pid):
        patch_id = f'r{rid}_{self.module_id}_p{pid}'
        # print(f'Writing patch {patch_id} for {self.model_path}')
        # self.print()
        # copy a new file of the model
        if os.path.exists(f'{self.model_path}.ori'):
            os.remove(f'{self.model_path}.ori')
        if os.path.exists(f'{self.model_path}.opt_{pid}'):
            os.remove(f'{self.model_path}.opt_{pid}')
        if os.path.exists(f'{self.model_dir}/{patch_id}.patch'):
            os.remove(f'{self.model_dir}/{patch_id}.patch')

        if not os.path.exists(f'{self.working_dir}/{RESULT_DIR}'):
            os.mkdir(f'{self.working_dir}/{RESULT_DIR}')
        if not os.path.exists(f'{self.working_dir}/{PATCH_DIR}'):
            os.mkdir(f'{self.working_dir}/{PATCH_DIR}')
        run_cmd(f'cp {self.model_path} {self.model_path}.ori')
        # read the original file in lines
        with open(self.model_path, 'r') as f:
            lines = f.readlines()
        # create a new empty file
        with open(f'{self.model_path}.opt_{pid}', 'w') as f:
            # for each line of the original file, if the line is in the affected lines, write the refactored code
            # otherwise, write the original code
            for i, line in enumerate(lines):
                if i + 1 in self.change_loc.keys():
                    # get the indent of this line in original file
                    indent = len(line) - len(line.lstrip())
                    # write the refactored code
                    trees = self.change_loc[i + 1].codes
                    for tree in trees:
                        try:
                            codes = astunparse.unparse(tree).strip().split('\n')
                        except:
                            a=1
                        for code in codes:
                            # write the indent and the code to the new file
                            f.write(' ' * indent + code + '\n')
                elif i + 1 in self.affected_lines:
                    continue
                else:
                    f.write(line)

        run_cmd(f'cd {self.working_dir}; diff -u {self.model_path}.ori {self.model_path}.opt_{pid} > {self.model_dir}/{patch_id}.patch')
        run_cmd(f'mv {self.model_path}.opt_{pid} {RESULT_DIR}')
        run_cmd(f'mv {self.model_dir}/{patch_id}.patch {PATCH_DIR}')
        if os.path.exists(f'{self.model_path}.ori'):
            os.remove(f'{self.model_path}.ori')