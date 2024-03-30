import ast, astunparse, importlib


class packageMngr(object):
    def __init__(self):
        self.alias = {}
        self.modules = {}
        self.tmp_alias = {}
        self.tmp_modules = {}

    def register(self, ast_node):
        if isinstance(ast_node, ast.Import):
            for name in ast_node.names:
                if name.asname is not None:
                    self.alias[name.asname] = name.name
        if isinstance(ast_node, ast.ImportFrom):
            for name in ast_node.names:
                if name.asname is not None:
                    self.alias[name.asname] = f'{ast_node.module}.{name.name}'
                else:
                    self.modules[name.name] = f'{ast_node.module}.{name.name}'

    def register_tmp(self, ast_node):
        if isinstance(ast_node, ast.Import):
            for name in ast_node.names:
                if name.asname is not None:
                    self.tmp_alias[name.asname] = name.name
        if isinstance(ast_node, ast.ImportFrom):
            for name in ast_node.names:
                if name.asname is not None:
                    self.tmp_alias[name.asname] = f'{ast_node.module}.{name.name}'
                else:
                    self.tmp_modules[name.name] = f'{ast_node.module}.{name.name}'

    def clear_tmp(self):
        self.tmp_alias = {}
        self.tmp_modules = {}

    def lookUp(self, module_id):
        if module_id in self.alias:
            return self.alias[module_id]
        elif module_id in self.modules:
            return self.modules[module_id]
        elif module_id in self.tmp_alias:
            return self.tmp_alias[module_id]
        elif module_id in self.tmp_modules:
            return self.tmp_modules[module_id]
        else:
            return None
