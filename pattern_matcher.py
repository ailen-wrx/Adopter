import ast, gast, astunparse, importlib, inspect
import copy
from enum import Enum

from python_graphs import control_flow
from package_mngr import packageMngr


FLG_BINOP_ADD = '#+'
FLG_TUPLE     = '#T'

LAYER = 1
BIN = 2
UNARY = 3


def unparse(node):
    source = astunparse.unparse(node)
    return (
        source.strip()
        .rstrip(' \n')
        .lstrip(' \n')
    )


def unparse_attribute(attribute):
    if isinstance(attribute, ast.Attribute):
        res = [attribute.attr]
        while isinstance(attribute.value, ast.Attribute):
            res = [attribute.value.attr] + res
            attribute = attribute.value
        if isinstance(attribute.value, ast.Attribute) or \
                isinstance(attribute.value, ast.Name):
            res = [attribute.value.id] + res
            return res
        elif isinstance(attribute.value, str):
            res = [attribute.value] + res
            return res
        else:
            res = [unparse(attribute.value)] + res
            return res
    elif isinstance(attribute, ast.Name):
        return [attribute.id]
    elif isinstance(attribute, ast.Tuple):
        attr = [FLG_TUPLE]
        for elt in attribute.elts:
            attr += unparse_attribute(elt)
        return attr
    elif isinstance(attribute, str):
        return [attribute]
    elif isinstance(attribute, ast.BinOp):
        if isinstance(attribute.op, ast.Add):
            return [FLG_BINOP_ADD, unparse_attribute(attribute.left), unparse_attribute(attribute.right)]

    return [unparse(attribute)]


def parse_attribute(attr):
    if attr[0] == FLG_BINOP_ADD:
        return f'{parse_attribute(attr[1])} + {parse_attribute((attr[2]))}'
    elif attr[0] == FLG_TUPLE:
        pass
    else:
        return '.'.join(attr)


class callStatement(object):
    def __init__(self, node, module_visitor, branch, seq, seq_assign=None, gast_node=None, parent=None, entry=None):
        self.global_visitor = module_visitor
        self.ast_node = node
        self.gast_node = gast_node
        self.output = None
        self.functionId = None
        self.refactoredFunctionId = None
        self.arguments = None
        self.keywords = None
        self.type = 0
        self.in_branch = False if branch is None else True
        self.branch = branch
        self.in_seq = False if seq is None else True
        self.seq = seq
        self.seq_assign = seq_assign
        self.nested = False if parent is None else True
        self.p = parent
        self.p_entry = entry

    def set_type(self, type):
        self.type = type

    def set_output(self, output):
        # if isinstance(output, ast.Tuple):
        #     pass
        self.output = unparse_attribute(output)

    def set_argument(self, arg):
        if not self.arguments:
            self.arguments = []
        self.arguments.append(arg)

    def set_keyword(self, kw):
        if not self.keywords:
            self.keywords = {}
        self.keywords[kw.arg] = unparse_attribute(kw.value)

    def set_function(self, func):
        if func is str:
            self.functionId = func
            return

        func_attributes = unparse_attribute(func)
        full_qual_func_id = self.global_visitor.package_mngr.lookUp(func_attributes[0])
        if full_qual_func_id is not None:
            func_attributes = full_qual_func_id.split('.') + func_attributes[1:]
        self.functionId = func_attributes

        # Refactoring
        if len(self.global_visitor.class_def_stack) == 0:
            if self.global_visitor.current_func.name in self.global_visitor.global_func_var_def and \
                    '.'.join(func_attributes) in self.global_visitor.global_func_var_def[self.global_visitor.current_func.name]:
                alias = self.global_visitor.global_func_var_def[self.global_visitor.current_func.name]['.'.join(func_attributes)]
                self.refactoredFunctionId['.'.join(alias)] = None
        else:
            class_info = self.global_visitor.class_def_stack[-1]
            if func_attributes[0] == 'self':
                alias = class_info.look_up_attribute('.'.join(func_attributes[1:]))
                if alias:
                    self.refactoredFunctionId = alias
            else:
                alias = class_info.look_up_variable(self.global_visitor.current_func.name, '.'.join(func_attributes))
                if alias:
                    self.refactoredFunctionId = alias

        # Save variables
        if self.output is not None:
            if len(self.global_visitor.class_def_stack) == 0:
                if self.global_visitor.current_func.name not in self.global_visitor.global_func_var_def:
                    self.global_visitor.global_func_var_def[self.global_visitor.current_func.name] = {}
                self.global_visitor.global_func_var_def[self.global_visitor.current_func.name][
                    '.'.join(self.output)] = self.functionId
            else:
                class_info = self.global_visitor.class_def_stack[-1]
                if self.output[0] == 'self':
                    class_info.set_attribute('.'.join(self.output[1:]), self.functionId, self)
                else:
                    class_info.set_variable(self.global_visitor.current_func.name, '.'.join(self.output),
                                            self.functionId, self)


class callVisitor(ast.NodeVisitor):
    def __init__(self, node, module_visitor, gast_node=None, inline_mode=False, inline_dict=None, branch=None, seq=None, seq_assign=None, parent=None, entry=None):
        self.global_visitor = module_visitor
        self.root = node
        self.gast_node = gast_node
        self.inline_mode = inline_mode
        self.inline_dict = inline_dict
        self.branch = branch
        self.seq = seq
        self.seq_assign = seq_assign
        self.parent = parent
        self.entry = entry
        self.temp_call_stmt = callStatement(node, module_visitor, gast_node=self.gast_node, branch=self.branch, seq=self.seq, seq_assign=self.seq_assign, parent=self.parent, entry=self.entry)

    def debug(self, node):
        print(unparse(node))

    def dump_call_stmt(self):
        return self.temp_call_stmt

    def visit_Import(self, node):
        self.global_visitor.package_mngr.register_tmp(node)

    def is_tensor_operation(self, node):
        if isinstance(node, ast.BinOp):
            return True
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.operand, ast.Constant):
                return False
            return True
        return False

    def visit_ImportFrom(self, node):
        self.global_visitor.package_mngr.register_tmp(node)

    def visit_Call(self, node):
        # self.debug(node)
        self.temp_call_stmt.set_type(LAYER)

        if self.inline_mode and isinstance(node.func, ast.Attribute) \
                and isinstance(node.func.value, ast.Name) \
                and node.func.value.id == 'self' \
                and node.func.attr in self.global_visitor.call_sequences[self.global_visitor.current_class]:
            function = self.global_visitor.call_sequences[self.global_visitor.current_class][node.func.attr]
            args = {}
            first_self = 0
            for i, arg in enumerate(function['_ast_node'].args.args):
                if arg.arg == 'self':
                    first_self = 1
                    continue
                matched = False
                for kw in node.keywords:
                    if arg.arg == kw.arg:
                        call_visitor = callVisitor(kw.value, self.global_visitor, inline_mode=True)
                        tmp_variable = self.global_visitor.get_temp_variable()
                        call_visitor.temp_call_stmt.set_output(tmp_variable)
                        call_visitor.visit(kw.value)
                        args[arg.arg] = kw.value
                        matched = True
                        break
                if not matched and i - first_self < len(node.args):
                    if isinstance(node.args[i - first_self], ast.Call):
                        call_visitor = callVisitor(node.args[i - first_self], self.global_visitor)
                        tmp_variable = self.global_visitor.get_temp_variable()
                        call_visitor.temp_call_stmt.set_output(tmp_variable)
                        call_visitor.visit(node.args[i - first_self])
                        args[arg.arg] = tmp_variable

            for call in function['_call_seqs'][0].call_statements:
                call_visitor = callVisitor(call.ast_node, self.global_visitor, inline_mode=True, inline_dict=args)
                call_visitor.visit(call.ast_node)

        if isinstance(node.func, ast.Attribute) and unparse_attribute(node.func)[-1] == 'Sequential':
            for i, arg in enumerate(node.args):
                if isinstance(arg, ast.Call):
                    call_visitor = callVisitor(arg, self.global_visitor, seq=node, seq_assign=self.root)
                    tmp_variable = 'sequential_' + self.global_visitor.get_temp_variable()
                    call_visitor.temp_call_stmt.set_output(tmp_variable)
                    call_visitor.visit(arg)
                    self.temp_call_stmt.set_argument([tmp_variable])
            self.temp_call_stmt.set_function(node.func)
            self.global_visitor.tmp_call_sequence.append_stmt(self)
            return

        for arg in node.args:
            if isinstance(arg, ast.Call) or self.is_tensor_operation(arg):
                call_visitor = callVisitor(arg, self.global_visitor, parent=self.temp_call_stmt, entry='arg')
                tmp_variable = self.global_visitor.get_temp_variable(call_visitor.dump_call_stmt())
                call_visitor.temp_call_stmt.set_output(tmp_variable)
                call_visitor.visit(arg)
                self.temp_call_stmt.set_argument([tmp_variable])
            else:
                self.temp_call_stmt.set_argument(unparse_attribute(arg))
        for kw in node.keywords:
            if isinstance(kw.value, ast.Call) or self.is_tensor_operation(kw.value):
                call_visitor = callVisitor(kw.value, self.global_visitor, parent=self.temp_call_stmt, entry='kw.value')
                tmp_variable = self.global_visitor.get_temp_variable(call_visitor.dump_call_stmt())
                call_visitor.temp_call_stmt.set_output(tmp_variable)
                call_visitor.visit(kw.value)
                kw1 = copy.copy(kw)
                kw1.value = ast.Name(tmp_variable)
                self.temp_call_stmt.set_keyword(kw1)
            else:
                self.temp_call_stmt.set_keyword(kw)
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Call):
            call_visitor = callVisitor(node.func.value, self.global_visitor, parent=self.temp_call_stmt, entry='node.func.value')
            tmp_variable = self.global_visitor.get_temp_variable(call_visitor.dump_call_stmt())
            call_visitor.temp_call_stmt.set_output(tmp_variable)
            call_visitor.visit(node.func.value)
            func = copy.copy(node.func)
            func.value = ast.Name(tmp_variable)
            self.temp_call_stmt.set_function(func)
        else:
            self.temp_call_stmt.set_function(node.func)
        # if isinstance(self.root, ast.Return):
        #     self.global_visitor.call_sequences[self.global_visitor.current_class]\
        #         [self.global_visitor.current_func.name]['_return'] = node
        # else:
        self.global_visitor.tmp_call_sequence.append_stmt(self)

    def visit_BinOp(self, node):
        self.temp_call_stmt.set_type(BIN)
        self.temp_call_stmt.functionId = [type(node.op).__name__]
        self.temp_call_stmt.set_argument(unparse_attribute(node.left))
        self.temp_call_stmt.set_argument(unparse_attribute(node.right))
        self.global_visitor.tmp_call_sequence.append_stmt(self)

    def visit_Assign(self, node):
        # self.debug(node)
        if self.inline_dict and node.targets[0] in self.inline_dict:
            self.temp_call_stmt.set_output(self.inline_dict[node.targets[0]])
        else:
            self.temp_call_stmt.set_output(node.targets[0])
        self.visit(node.value)

    def visit_Subscript(self, node):
        self.temp_call_stmt.set_function(astunparse.unparse(node).strip())
        self.global_visitor.tmp_call_sequence.append_stmt(self)

    def visit_Attribute(self, node):
        self.temp_call_stmt.set_function(astunparse.unparse(node).strip())
        self.global_visitor.tmp_call_sequence.append_stmt(self)


class callSequence(object):
    def __init__(self, module_visitor):
        self.global_visitor = module_visitor
        self.call_statements = []
        self.function_def = module_visitor.current_func
        self.class_def = copy.copy(module_visitor.class_def_stack)

    def append_stmt(self, call_visitor):
        stmt = call_visitor.dump_call_stmt()
        self.call_statements.append(stmt)

    def print(self, i=0, j=-1):
        linenums = []
        codes = []
        if j == -1:
            j = len(self.call_statements)
        if set([call_stmt.in_seq for call_stmt in self.call_statements[i: j]]) == {True}:
            nn_seq = self.call_statements[i].seq
            if nn_seq.end_lineno:
                for i in range(self.function_def.lineno + nn_seq.lineno - 1, self.function_def.lineno + nn_seq.end_lineno):
                    linenums.append(i)
            else:
                linenums.append(self.function_def.lineno + nn_seq.lineno - 1)
            codes.append(f'{astunparse.unparse(nn_seq).strip()}')
            return linenums, codes

        for idx, stmt0 in enumerate(self.call_statements[i: j]):
            stmt = copy.copy(stmt0)
            if stmt.in_branch and idx > 0 and not self.call_statements[i + idx - 1].in_branch:
                linenums.clear()
                codes.clear()
            while stmt.nested:
                stmt = stmt.p
            code = astunparse.unparse(stmt.ast_node)
            codes.append(f'{code.strip()}')
            linenums.append(self.function_def.lineno + stmt.ast_node.lineno - 1)
        return linenums, codes


    def get_trees_from_index(self, idx_list):
        call_statements = [self.call_statements[idx] for idx in idx_list]
        linenums = []
        trees = []
        entry = None
        cpy_entry= None
        entry_idx = []
        _inter_branch = set([call_stmt.in_branch for call_stmt in call_statements])
        inter_branch = True in _inter_branch and False in _inter_branch
        branch = None
        have_branch = False
        for idx, stmt0 in enumerate(call_statements):
            stmt = copy.copy(stmt0)
            while stmt.nested:
                stmt = stmt.p

            branch_offset = 0
            if inter_branch:
                if stmt.in_branch and stmt.gast_node:
                    if not have_branch:
                        branch_rdonly = stmt.branch
                        branch = copy.deepcopy(stmt.branch)
                        have_branch = True
                        entry = stmt.branch.body if stmt.gast_node in stmt.branch.body else stmt.branch.orelse
                        cpy_entry = branch.body if stmt.gast_node in stmt.branch.body else branch.orelse
                        if branch.end_lineno:
                            for i in range(self.function_def.lineno + branch.lineno - 1,
                                           self.function_def.lineno + branch.end_lineno):
                                linenums.append(i)
                        else:
                            linenums.append(self.function_def.lineno + branch.lineno - 1)
                        for i_tree, tree in enumerate(trees):
                            branch.body.insert(i_tree, tree)
                            branch.orelse.insert(i_tree, tree)
                            branch_offset += 1
                            entry_idx.append(i_tree)
                        entry_idx.append(entry.index(stmt.gast_node) + branch_offset)
                        trees.clear()
                        trees.append(branch)
                    elif stmt.branch == branch_rdonly:
                        entry_idx.append(entry.index(stmt.gast_node) + branch_offset)
                    else:
                        pass

                elif have_branch and stmt.gast_node:
                    branch.body.insert(len(branch.body), stmt.gast_node)
                    branch.orelse.insert(len(branch.body), stmt.gast_node)
                    entry_idx.append(cpy_entry.index(stmt.gast_node))
                    if stmt.ast_node.end_lineno:
                        for i in range(self.function_def.lineno + stmt.ast_node.lineno - 1,
                                       self.function_def.lineno + stmt.ast_node.end_lineno):
                            linenums.append(i)
                    elif stmt.ast_node.value.end_lineno:
                        for i in range(self.function_def.lineno + stmt.ast_node.value.lineno - 1,
                                       self.function_def.lineno + stmt.ast_node.value.end_lineno):
                            linenums.append(i)
                    else:
                        linenums.append(self.function_def.lineno + stmt.ast_node.lineno - 1)

                elif stmt.gast_node:
                    trees.append(stmt.gast_node)
                    if stmt.ast_node.end_lineno:
                        for i in range(self.function_def.lineno + stmt.ast_node.lineno - 1,
                                       self.function_def.lineno + stmt.ast_node.end_lineno):
                            linenums.append(i)
                    elif stmt.ast_node.value.end_lineno:
                        for i in range(self.function_def.lineno + stmt.ast_node.value.lineno - 1,
                                       self.function_def.lineno + stmt.ast_node.value.end_lineno):
                            linenums.append(i)
                    else:
                        linenums.append(self.function_def.lineno + stmt.ast_node.lineno - 1)
                else:
                    trees.append(stmt.ast_node)
                    if stmt.ast_node.end_lineno:
                        for i in range(self.function_def.lineno + stmt.ast_node.lineno - 1,
                                       self.function_def.lineno + stmt.ast_node.end_lineno):
                            linenums.append(i)
                    elif stmt.ast_node.value.end_lineno:
                        for i in range(self.function_def.lineno + stmt.ast_node.value.lineno - 1,
                                       self.function_def.lineno + stmt.ast_node.value.end_lineno):
                            linenums.append(i)
                    else:
                        linenums.append(self.function_def.lineno + stmt.ast_node.lineno - 1)

            else:
                if stmt.gast_node:
                    trees.append(stmt.gast_node)
                    if stmt.ast_node.end_lineno:
                        for i in range(self.function_def.lineno + stmt.ast_node.lineno - 1,
                                       self.function_def.lineno + stmt.ast_node.end_lineno):
                            linenums.append(i)
                    elif stmt.ast_node.value.end_lineno:
                        for i in range(self.function_def.lineno + stmt.ast_node.value.lineno - 1,
                                       self.function_def.lineno + stmt.ast_node.value.end_lineno):
                            linenums.append(i)
                    else:
                        linenums.append(self.function_def.lineno + stmt.ast_node.lineno - 1)
                else:
                    trees.append(stmt.ast_node)
                    if stmt.ast_node.end_lineno:
                        for i in range(self.function_def.lineno + stmt.ast_node.lineno - 1,
                                       self.function_def.lineno + stmt.ast_node.end_lineno):
                            linenums.append(i)
                    elif stmt.ast_node.value.end_lineno:
                        for i in range(self.function_def.lineno + stmt.ast_node.value.lineno - 1,
                                       self.function_def.lineno + stmt.ast_node.value.end_lineno):
                            linenums.append(i)
                    else:
                        linenums.append(self.function_def.lineno + stmt.ast_node.lineno - 1)
        if not cpy_entry:
            cpy_entry = trees
            for i_tree, tree in enumerate(trees):
                entry_idx.append(i_tree)
        return linenums, trees, cpy_entry, entry_idx

    def get_trees(self, ii=0, jj=-1):
        linenums = []
        trees = []
        entry = None
        cpy_entry= None
        entry_idx = []
        _inter_branch = set([call_stmt.in_branch for call_stmt in self.call_statements[ii: jj]])
        inter_branch = True in _inter_branch and False in _inter_branch
        if jj == -1:
            jj = len(self.call_statements)
        if set([call_stmt.in_seq for call_stmt in self.call_statements[ii: jj]]) == {True}:
            nn_seq = self.call_statements[ii].seq_assign
            nn_seq_call = self.call_statements[ii].seq
            if nn_seq.end_lineno:
                for i in range(self.function_def.lineno + nn_seq.lineno - 1, self.function_def.lineno + nn_seq.end_lineno):
                    linenums.append(i)
            elif nn_seq_call.end_lineno:
                for i in range(self.function_def.lineno + nn_seq_call.lineno - 1, self.function_def.lineno + nn_seq_call.end_lineno):
                    linenums.append(i)
            else:
                linenums.append(self.function_def.lineno + nn_seq.lineno - 1)
            trees.append(nn_seq)
            entry = trees
            entry_idx = [0]
            return linenums, trees, entry, entry_idx


        branch = None
        have_branch = False
        for idx, stmt0 in enumerate(self.call_statements[ii: jj]):
            stmt = copy.copy(stmt0)
            while stmt.nested:
                stmt = stmt.p

            branch_offset = 0
            if inter_branch:
                if stmt.in_branch and stmt.gast_node:
                    if not have_branch:
                        branch_rdonly = stmt.branch
                        branch = copy.deepcopy(stmt.branch)
                        have_branch = True
                        entry = stmt.branch.body if stmt.gast_node in stmt.branch.body else stmt.branch.orelse
                        cpy_entry = branch.body if stmt.gast_node in stmt.branch.body else branch.orelse
                        if branch.end_lineno:
                            for i in range(self.function_def.lineno + branch.lineno - 1,
                                           self.function_def.lineno + branch.end_lineno):
                                linenums.append(i)
                        else:
                            linenums.append(self.function_def.lineno + branch.lineno - 1)
                        for i_tree, tree in enumerate(trees):
                            branch.body.insert(i_tree, tree)
                            branch.orelse.insert(i_tree, tree)
                            branch_offset += 1
                            entry_idx.append(i_tree)
                        entry_idx.append(entry.index(stmt.gast_node) + branch_offset)
                        trees.clear()
                        trees.append(branch)
                    elif stmt.branch == branch_rdonly:
                        entry_idx.append(entry.index(stmt.gast_node) + branch_offset)
                    else:
                        pass

                elif have_branch and stmt.gast_node:
                    branch.body.insert(len(branch.body), stmt.gast_node)
                    branch.orelse.insert(len(branch.body), stmt.gast_node)
                    entry_idx.append(cpy_entry.index(stmt.gast_node))
                    if stmt.ast_node.end_lineno:
                        for i in range(self.function_def.lineno + stmt.ast_node.lineno - 1,
                                       self.function_def.lineno + stmt.ast_node.end_lineno):
                            linenums.append(i)
                    elif stmt.ast_node.value.end_lineno:
                        for i in range(self.function_def.lineno + stmt.ast_node.value.lineno - 1,
                                       self.function_def.lineno + stmt.ast_node.value.end_lineno):
                            linenums.append(i)
                    else:
                        linenums.append(self.function_def.lineno + stmt.ast_node.lineno - 1)

                if stmt.gast_node:
                    trees.append(stmt.gast_node)
                    if stmt.ast_node.end_lineno:
                        for i in range(self.function_def.lineno + stmt.ast_node.lineno - 1,
                                       self.function_def.lineno + stmt.ast_node.end_lineno):
                            linenums.append(i)
                    elif stmt.ast_node.value.end_lineno:
                        for i in range(self.function_def.lineno + stmt.ast_node.value.lineno - 1,
                                       self.function_def.lineno + stmt.ast_node.value.end_lineno):
                            linenums.append(i)
                    else:
                        linenums.append(self.function_def.lineno + stmt.ast_node.lineno - 1)
                else:
                    trees.append(stmt.ast_node)
                    if stmt.ast_node.end_lineno:
                        for i in range(self.function_def.lineno + stmt.ast_node.lineno - 1,
                                       self.function_def.lineno + stmt.ast_node.end_lineno):
                            linenums.append(i)
                    elif stmt.ast_node.value.end_lineno:
                        for i in range(self.function_def.lineno + stmt.ast_node.value.lineno - 1,
                                       self.function_def.lineno + stmt.ast_node.value.end_lineno):
                            linenums.append(i)
                    else:
                        linenums.append(self.function_def.lineno + stmt.ast_node.lineno - 1)

            else:
                if stmt.gast_node:
                    trees.append(stmt.gast_node)
                    if stmt.ast_node.end_lineno:
                        for i in range(self.function_def.lineno + stmt.ast_node.lineno - 1,
                                       self.function_def.lineno + stmt.ast_node.end_lineno):
                            linenums.append(i)
                    elif stmt.ast_node.value.end_lineno:
                        for i in range(self.function_def.lineno + stmt.ast_node.value.lineno - 1,
                                       self.function_def.lineno + stmt.ast_node.value.end_lineno):
                            linenums.append(i)
                    else:
                        linenums.append(self.function_def.lineno + stmt.ast_node.lineno - 1)
                else:
                    trees.append(stmt.ast_node)
                    if stmt.ast_node.end_lineno:
                        for i in range(self.function_def.lineno + stmt.ast_node.lineno - 1,
                                       self.function_def.lineno + stmt.ast_node.end_lineno):
                            linenums.append(i)
                    elif stmt.ast_node.value.end_lineno:
                        for i in range(self.function_def.lineno + stmt.ast_node.value.lineno - 1,
                                       self.function_def.lineno + stmt.ast_node.value.end_lineno):
                            linenums.append(i)
                    else:
                        linenums.append(self.function_def.lineno + stmt.ast_node.lineno - 1)
        if not cpy_entry:
            cpy_entry = trees
            for i_tree, tree in enumerate(trees):
                entry_idx.append(i_tree)
        return linenums, trees, cpy_entry, entry_idx

class ClassInfo(object):
    def __init__(self, node, func_dict, class_dict):
        self.node = node
        self.functions = func_dict
        self.subclasses = class_dict
        self.attributes = {}
        self.variable_def = {}

    def set_attribute(self, attr, node, stmt):
        if attr not in self.attributes:
            self.attributes[attr] = {}
        self.attributes[attr]['.'.join(node)] = {
            'id': node,
            'stmt': stmt
        }

    def set_variable(self, func_name, var, node, stmt):
        if func_name not in self.variable_def:
            self.variable_def[func_name] = {}
        if var not in self.variable_def[func_name]:
            self.variable_def[func_name][var] = {}
        self.variable_def[func_name][var]['.'.join(node)] = {
            'id': node,
            'stmt': stmt
        }

    def look_up_attribute(self, attr):
        if attr in self.attributes:
            return self.attributes[attr]
        else:
            return None

    def look_up_variable(self, func_name, var):
        if func_name not in self.variable_def:
            return None
        elif var in self.variable_def[func_name]:
            return self.variable_def[func_name][var]
        else:
            return None


class ModelVisitor(ast.NodeVisitor):
    def __init__(self, module, ablation):
        self.model_name = module
        self.model_path = module.replace('.', '/') + '.py'
        self.package_mngr = packageMngr()
        self.global_var = {}
        self.global_func_var_def = {}
        tc = importlib.import_module(module)
        self.dict = tc.__dict__

        self.call_sequences = {}
        self.call_sequences_in_function = []
        self.class_def_stack = []
        self.current_class = None
        self.current_func = None
        self.temp_var_index = 0
        self.temp_var_dict = {}
        self.tmp_call_sequence = None
        self.ablation = ablation

        with open(self.model_path) as f:
            self.ast_node = ast.parse(f.read())
        self.ast_dict = self.get_ast_dict(self.ast_node)


    def get_ast_dict(self, node):
        ast_dict = {'import': []}
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef):
                ast_dict[child.name] = child
            elif isinstance(child, ast.ClassDef):
                ast_dict[child.name] = {}
                for child1 in ast.iter_child_nodes(child):
                    if isinstance(child1, ast.FunctionDef):
                        ast_dict[child.name][child1.name] = child1
            elif isinstance(child, ast.Import):
                ast_dict['import'].append(child)
        return ast_dict

    def get_paths(self, graph):
        init_node = None
        for block in graph.blocks:
            init_idx = 0
            if block.label is not None and block.label.startswith('<'):
                continue
            if not block.control_flow_nodes or len(block.control_flow_nodes) == 0:
                continue
            if isinstance(block.control_flow_nodes[init_idx].instruction.node, gast.gast.arguments):
                init_idx += 1
            try:
                astunparse.unparse(block.control_flow_nodes[init_idx].instruction.node)
                if astunparse.unparse(block.control_flow_nodes[init_idx].instruction.node).strip() == 'super().__init__()':
                    init_idx += 1
                init_node = block.control_flow_nodes[init_idx]
                break
            except:
                continue
        if not init_node:
            return None
        return self._get_paths(init_node, [], [0])

    def _get_paths(self, node, visited_nodes, num_paths):
        if len(node.next) == 0 or num_paths[0] > 20:
            num_paths[0] += 1
            return [[node]]
        paths = []
        for next_node in node.next:
            if next_node in visited_nodes:
                continue
            for path in self._get_paths(next_node, visited_nodes + [node], num_paths):
                paths.append([node] + path)
        return paths

    def get_blocks(self, graph):
        blocks = []
        for block in graph.blocks:
            b = []
            if block.label is not None and block.label.startswith('<'):
                continue
            if not block.control_flow_nodes or len(block.control_flow_nodes) == 0:
                continue
            for node in block.control_flow_nodes:
                if isinstance(node.instruction.node, gast.gast.arguments):
                    continue
                try:
                    astunparse.unparse(node.instruction.node)
                    if astunparse.unparse(node.instruction.node).strip() == 'super().__init__()':
                        continue
                except:
                    continue
                b.append(copy.deepcopy(node))
            blocks.append(b)
        return blocks

    def get_temp_variable(self, stmt=None):
        var_name = f'@@temp_{self.temp_var_index}'
        self.temp_var_dict[var_name] = stmt
        self.temp_var_index += 1
        return var_name

    def begin_visit(self):
        self.visit(self.ast_node)
        del self.call_sequences_in_function, self.class_def_stack, \
            self.current_class, self.current_func, self.tmp_call_sequence, \
            self.temp_var_index, self.dict

    def visit_Import(self, node):
        self.package_mngr.register(node)

    def visit_ImportFrom(self, node):
        self.package_mngr.register(node)

    def visit_Assign(self, node):
        if len(self.class_def_stack) != 0:
            class_info = self.class_def_stack[-1]
            class_info.set_variable('__self__', unparse(node.targets[0]), unparse(node.value).split('.'), self)
            return
        self.global_var[unparse(node.targets[0])] = node.value

    def visit_ClassDef(self, node):
        # print("[debug] Visit class " + node.name)
        class_name = node.name
        if len(self.class_def_stack) == 0:
            class_inspectee = self.dict[class_name]
        else:
            class_inspectee = self.class_def_stack[-1].subclasses[class_name]
        func_dict = {}
        class_dict = {}
        for fname, fn in inspect.getmembers(class_inspectee, predicate=inspect.isfunction):
            func_dict[fname] = fn
        for cname, cls in inspect.getmembers(class_inspectee, predicate=inspect.isclass):
            class_dict[cname] = cls
        self.class_def_stack.append(ClassInfo(node, func_dict, class_dict))
        self.current_class = '.'.join([classdef.node.name for classdef in self.class_def_stack])
        self.call_sequences[self.current_class] = {}
        self.generic_visit(node)
        self.class_def_stack.pop()

    def visit_FunctionDef(self, node):
        # print("[debug] Visit function " + node.name)
        function_name = node.name
        if len(self.class_def_stack) != 0 and function_name in self.class_def_stack[-1].functions:
            function_inspector = self.class_def_stack[-1].functions[function_name]
        else:
            return
        graph = control_flow.get_control_flow_graph(function_inspector)
        self.current_func = node
        self.call_sequences_in_function = []

        # paths = self.get_paths(graph)
        if function_name == 'forward' or function_name == '__init__':
            if self.ablation == "path":
                paths = self.get_blocks(graph)
            else:
                try:
                    paths = self.get_paths(graph)
                except:
                    return
        else:
            paths = self.get_blocks(graph)
        if paths is None:
            self.current_func = None
            return

        # self.call_sequences[self.current_class] = []
        self.call_sequences[self.current_class][self.current_func.name] = {
            '_ast_node': node
        }

        for path in paths:
            self.tmp_call_sequence = callSequence(self)
            for control_flow_node in path:
                instruction_node = control_flow_node.instruction.node
                ast_node = gast.gast_to_ast(instruction_node)
                try:
                    branch = control_flow_node.block.node
                    assert isinstance(branch, gast.gast.If) and instruction_node.lineno in range(branch.lineno, branch.end_lineno + 1)
                    call_visitor = callVisitor(ast_node, self, gast_node=instruction_node, branch=branch)
                except:
                    call_visitor = callVisitor(ast_node, self, gast_node=instruction_node)
                if isinstance(ast_node, ast.Assign):
                    call_visitor.visit(ast_node)
            if len(self.tmp_call_sequence.call_statements) > 0:
                self.call_sequences_in_function.append(self.tmp_call_sequence)
                # self.tmp_call_sequence.print()
            pass

        self.package_mngr.clear_tmp()
        # self.call_sequences[self.current_class] = self.call_sequences[self.current_class] + self.call_sequences_in_function
        self.call_sequences[self.current_class][self.current_func.name]['_call_seqs'] = self.call_sequences_in_function
        self.current_func = None
        return


if __name__ == '__main__':
    test = ModelVisitor('models.huggingface_transformers.src.transformers.models.bert.modeling_bert', None)
    test.begin_visit()
