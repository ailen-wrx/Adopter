import ast, gast, astunparse, importlib, inspect
import copy

from python_graphs import control_flow
from package_mngr import packageMngr


def unparse(node):
    source = astunparse.unparse(node)
    return (
        source.strip()
        .rstrip(' \n')
        .lstrip(' \n')
    )

class callStatement(object):
    def __init__(self, inst_node, function_ipa):
        self.Function = function_ipa
        self.Instruction = inst_node
        self.unimplemented = False

        # extract output
        assert len(inst_node.targets) == 1
        output = self.Function.unparse_attribute(inst_node.targets[0])
        if not output:
            # TODO: Did not implement for stmts like 'x, y = func()'
            self.unimplemented = True
            return
        self.output = '.'.join(output)

        # extract function id
        pkgMngr = self.Function.Class.Module.package_mngr
        call = inst_node.value
        func = self.Function.unparse_attribute(call.func)
        if not func:
            # TODO: Did not implement for nested call f(g(x,y))
            self.unimplemented = True
            return
        function_refactor = None
        if '.'.join(func) in self.Function.variables:
            function_refactor = self.Function.variables['.'.join(func)]
        elif func[0] == 'self' and '.'.join(func[1:]) in self.Function.Class.attributes:
            function_refactor = self.Function.Class.attributes['.'.join(func[1:])]
        self.refactoredFunctionId = None if not function_refactor \
            else self.Function.unparse_attribute(function_refactor.func)

        full_qual_func_id = pkgMngr.lookUp(func[0])
        if full_qual_func_id is not None:
            full_qual_func_id = full_qual_func_id.split('.') + func[1:]
            self.functionId = full_qual_func_id
        else:
            self.functionId = func

        if self.refactoredFunctionId is not None:
            full_qual_func_refactor_id = pkgMngr.lookUp(self.refactoredFunctionId[0])
            if full_qual_func_refactor_id is not None:
                full_qual_func_refactor_id = full_qual_func_refactor_id.split('.') + self.refactoredFunctionId[1:]
                self.refactoredFunctionId = full_qual_func_refactor_id
            else:
                self.refactoredFunctionId = self.refactoredFunctionId

        # extract args
        self.arguments = []
        self.keywords = []
        for arg in call.args:
            self.arguments.append(unparse(arg))
        for kwarg in call.keywords:
            self.keywords.append({
                'key': kwarg.arg,
                'arg': unparse(kwarg.value)
            })


class callSequence(object):
    def __init__(self, function_ipa):
        self.Function = function_ipa
        self.call_statements = []

    def append_stmt(self, inst_node):
        self.call_statements.append(callStatement(inst_node, self.Function))

    def print(self, i, j):
        print(f'Class: {self.Function.Class.name}, Function: {self.Function.name}')
        for stmt in self.call_statements[i: j]:
            print(f'{self.Function.ast_node.lineno + stmt.Instruction.lineno - 1}    {unparse(stmt.Instruction)}')
        print('')


class functionIPA(ast.NodeVisitor):
    def get_paths(self, node):
        if len(node.next) == 0:
            return [[node]]
        paths = []
        for next_node in node.next:
            for path in self.get_paths(next_node):
                paths.append([node] + path)
        return paths

    def unparse_attribute(self, attribute):
        if isinstance(attribute, gast.gast.Attribute):
            res = [attribute.attr]
            while isinstance(attribute.value, gast.gast.Attribute):
                res = [attribute.value.attr] + res
                attribute = attribute.value
            if isinstance(attribute.value, gast.gast.Attribute) or \
               isinstance(attribute.value, gast.gast.Name):
                res = [attribute.value.id] + res
            else:
                # TODO: Did not implement for nested call f(g(x,y))
                return []
            return res
        elif isinstance(attribute, gast.gast.Name):
            return [attribute.id]

    def __init__(self, ast_node, fn_obj, class_ipa):
        self.name = ast_node.name
        self.ast_node = ast_node
        self.variables = {}
        self.Class = class_ipa
        self.call_sequences = []
        graph = control_flow.get_control_flow_graph(fn_obj)

        init_node = None
        for block in graph.blocks:
            if block.label is not None and block.label.startswith('<'):
                continue
            if not block.control_flow_nodes or len(block.control_flow_nodes) == 0: continue
            init_node = block.control_flow_nodes[0]
            break
        if not init_node:
            return

        paths = self.get_paths(init_node)

        blocks = []
        for block in graph.blocks:
            if block.label is not None and block.label.startswith('<'):
                continue
            if not block.control_flow_nodes or len(block.control_flow_nodes) == 0:
                continue
            blocks.append(block.control_flow_nodes)

        for path in paths:
            call_sequence = callSequence(self)
            for control_flow_node in path:
                instruction = control_flow_node.instruction
                node = instruction.node

                if isinstance(node, gast.gast.Assign) and isinstance(node.value, gast.gast.Call):
                    assert len(node.targets) == 1
                    for target in node.targets:
                        attributes = self.unparse_attribute(target)
                        if self.name == '__init__' and attributes[0] == 'self':
                            self.Class.attributes['.'.join(attributes[1:])] = node.value
                        else:
                            try:
                                self.variables['.'.join(attributes)] = node.value
                            except Exception:
                                # TODO: Did not implement for stmts like 'x, y = func()'
                                continue

                    call_sequence.append_stmt(node)

            self.Class.Module.call_sequences.append(call_sequence)
            self.call_sequences.append(call_sequence)


class classIPA(object):
    def __init__(self, ast_node, class_obj, module_ipa):
        self.name = ast_node.name
        self.ast_node = ast_node
        self.attributes = {}
        self.functions = []
        self.Module = module_ipa
        class_dict = {}
        for fname, fn in inspect.getmembers(class_obj, predicate=inspect.isfunction):
            class_dict[fname] = fn
        for class_member in self.ast_node.body:
            if not isinstance(class_member, ast.FunctionDef): continue
            fname = class_member.name
            fn = class_dict[fname]
            self.functions.append(functionIPA(class_member, fn, self))


class moduleIPA(ast.NodeVisitor):
    def __init__(self, module):
        self.module_id = module
        self.module_path = module.replace('.', '/') + '.py'
        self.packageMngr = packageMngr()
        self.global_var = {}
        self.call_sequences = []

        tc = importlib.import_module(module)
        self.module_dict = tc.__dict__
        self.classes = []

        with open(self.module_path) as f:
            ast_nodes = ast.parse(f.read())

        for ast_node in ast_nodes.body:
            if isinstance(ast_node, ast.Import) or isinstance(ast_node, ast.ImportFrom):
                self.packageMngr.register(ast_node)
            if isinstance(ast_node, ast.Assign):
                for target in ast_node.targets:
                    self.global_var[unparse(target)] = ast_node.value
            elif isinstance(ast_node, ast.FunctionDef):
                pass
            elif isinstance(ast_node, ast.ClassDef):
                cname = ast_node.name
                cls = self.module_dict[cname]
                self.classes.append(classIPA(ast_node, cls, self))


