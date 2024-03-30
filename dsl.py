import ast, astunparse, re
import copy

import gast

from pattern_matcher import unparse_attribute


class refactorHelper(object):
    def __init__(self, rules):
        self.rules = []
        self.arg_dict = {}
        for rule in rules:
            self.rules.append(Rule(rule, self.arg_dict))

        dest_alias_map = {}
        if len(self.rules) > 1:
            rule0 = self.rules[0]
            rule1 = self.rules[1]
            for dest_pattern in rule1.destPatterns:
                if isinstance(dest_pattern.output, list):
                    continue
                else:
                    dest_alias_map[dest_pattern.functionCall.functionId.id] = dest_pattern.output.value
            for dest_pattern in rule0.destPatterns:
                if dest_pattern.functionCall.functionId.id in dest_alias_map:
                    if '$' in dest_alias_map[dest_pattern.functionCall.functionId.id]:
                        dest_pattern.functionCall.functionId.is_variable = True
                    dest_pattern.functionCall.functionId.id = dest_alias_map[dest_pattern.functionCall.functionId.id]

            for pi, src_pattern in enumerate(rule0.srcPatterns):
                if pi < len(rule1.srcPatterns):
                    src_pattern.functionCall.definition = rule1.srcPatterns[pi].functionCall
            for pi, dest_pattern in enumerate(rule0.destPatterns):
                if pi < len(rule1.destPatterns):
                    dest_pattern.functionCall.definition = rule1.destPatterns[pi].functionCall


class Rule(object):
    def __init__(self, rule, arg_dict):
        self.srcPatterns = []
        self.destPatterns = []
        self.arg_dict = arg_dict
        self.detect_only = False
        self.in_order = False

        src, dest = rule.split('=>')
        for src_assignment in src.split(';'):
            if src_assignment.strip() == "<any_assignment>":
                self.srcPatterns.append("<any_assignment>")
            else:
                assignment = Assignment(src_assignment.strip(), self)
                if assignment.functionCall.functionId.in_order:
                    self.in_order = True
                self.srcPatterns.append(assignment)
        if dest.strip() == "_":
            self.detect_only = True
            return
        for dest_assignment in dest.split(';'):
            self.destPatterns.append(Assignment(dest_assignment.strip(), self))

    def clear_arg_dict(self):
        for idx in self.arg_dict:
            self.arg_dict[idx] = None

    def setArg(self, arg, val):
        self.arg_dict[arg] = val

    def unorderedPatternMatch(self, call_statements, alias_map, ablation):
        if len(call_statements) < len(self.srcPatterns):
            return None, None
        match = self.unorderedSubSequenceMatch(call_statements, alias_map, ablation)
        if match:
            return match, copy.deepcopy(self.arg_dict)
        else:
            return None, None

    def patternMatch(self, call_statements):
        # self.clear_arg_dict()
        locations = []
        if len(call_statements) < len(self.srcPatterns):
            return locations
        for i in range(len(call_statements) - len(self.srcPatterns) + 1):
            if self.subSequenceMatch(call_statements[i: i + len(self.srcPatterns)]):
                locations.append((i, i + len(self.srcPatterns), copy.deepcopy(self.arg_dict)))
                return locations
        return locations

    def unorderedSubSequenceMatch(self, call_sequence, alias_map, ablation):
        idx_list = []
        for idx in self.arg_dict:
            self.arg_dict[idx] = None

        srcPatterns = copy.deepcopy(self.srcPatterns)
        for j in range(len(srcPatterns)):
            have_match = False
            src_pattern = srcPatterns[j]
            for i in range(len(call_sequence)):
                if isinstance(src_pattern, str):
                    have_match = True
                    idx_list.append(i)
                    break
                call_stmt = call_sequence[i]
                if call_stmt.refactoredFunctionId is None:
                    if src_pattern.functionCall.functionId.id.startswith('*'):
                        if src_pattern.functionCall.functionId.id[1:].lower() not in (
                                '.'.join(call_stmt.functionId)).lower():
                            continue
                    else:
                        if src_pattern.functionCall.functionId.id != '.'.join(call_stmt.functionId):
                            continue
                else:
                    continue

                _functionId = '.'.join(call_stmt.functionId)
                _alias = '.'.join(call_stmt.output)
                if (not _functionId in alias_map.keys()) or (not _alias in alias_map[_functionId]):
                    continue

                if not src_pattern.output.assigned and not call_stmt.in_seq:
                    idx = src_pattern.output.variable[0]
                    if idx not in self.arg_dict:
                        continue
                    else :
                        self.arg_dict[idx] = call_stmt.output

                arg_matched = True
                for arg_idx, arg in enumerate(src_pattern.functionCall.arguments):
                    arg_matched = False
                    if isinstance(arg, Argument) and arg.value == '':
                        continue
                    if not call_stmt.arguments or len(call_stmt.arguments) <= arg_idx:
                        try:
                            if not call_stmt.keywords or \
                                    len(list(call_stmt.keywords.keys())) + len(call_stmt.arguments) <= arg_idx:
                                continue
                            else:
                                stmt_arg = call_stmt.keywords[
                                    list(call_stmt.keywords.keys())[arg_idx - len(call_stmt.arguments)]]
                        except:
                            continue
                    else:
                        stmt_arg = call_stmt.arguments[arg_idx]
                    if isinstance(arg, list):
                        if arg[0] == '+' and stmt_arg[0] == '#+':

                            if not arg[1].assigned:
                                idx = arg[1].variable[0]
                                if idx not in self.arg_dict:
                                    continue
                                elif not self.arg_dict[idx]:
                                    self.arg_dict[idx] = stmt_arg[1]
                                elif self.arg_dict[idx] != stmt_arg[1]:
                                    continue

                            if not arg[2].assigned:
                                idx = arg[2].variable[0]
                                if idx not in self.arg_dict:
                                    continue
                                elif not self.arg_dict[idx]:
                                    self.arg_dict[idx] = stmt_arg[2]
                                elif self.arg_dict[idx] != stmt_arg[2]:
                                    continue

                    elif not arg.assigned:
                        idx = arg.variable[0]
                        if idx not in self.arg_dict:
                            continue
                        elif not self.arg_dict[idx]:
                            self.arg_dict[idx] = stmt_arg
                        elif self.arg_dict[idx] != stmt_arg:
                            continue

                    arg_matched = True

                kw_matched = True
                for kw in src_pattern.functionCall.keywords:
                    kw_matched = False
                    if call_stmt.keywords is None or kw.key not in call_stmt.keywords:
                        kw_matched = True
                        continue
                    stmt_kw_arg = call_stmt.keywords[kw.key]
                    if not kw.argument.assigned:
                        idx = kw.argument.variable[0]
                        if idx not in self.arg_dict:
                            continue
                        elif not self.arg_dict[idx]:
                            self.arg_dict[idx] = stmt_kw_arg
                        elif self.arg_dict[idx] != stmt_kw_arg:
                            continue
                    else:
                        stmt_kw_arg = call_stmt.keywords[kw.key][0].replace(" ", "")
                        if kw.argument.value not in stmt_kw_arg:
                            continue
                    kw_matched = True

                if arg_matched and kw_matched:
                    have_match = True
                    idx_list.append(i)
                # break
            if not have_match:
                return None
        return idx_list

    def subSequenceMatch(self, call_sequence):
        for idx in self.arg_dict:
            self.arg_dict[idx] = None

        srcPatterns = copy.deepcopy(self.srcPatterns)

        if set([call_stmt.in_seq for call_stmt in call_sequence]) == {True}:
            for srcPattern in srcPatterns:
                if srcPattern.functionCall.definition:
                    srcPattern.functionCall = srcPattern.functionCall.definition

        for i in range(len(srcPatterns)):
            src_pattern = srcPatterns[i]
            if isinstance(src_pattern, str):
                continue
            call_stmt = call_sequence[i]
            if call_stmt.refactoredFunctionId is None:
                if src_pattern.functionCall.functionId.id != '.'.join(call_stmt.functionId):
                    return 0
            else:
                if not src_pattern.functionCall.functionId.in_order or \
                        src_pattern.functionCall.functionId.id not in call_stmt.refactoredFunctionId:
                    return 0

            if not src_pattern.output.assigned and not call_stmt.in_seq:
                idx = src_pattern.output.variable[0]
                if idx not in self.arg_dict:
                    return 0
                elif not self.arg_dict[idx]:
                    self.arg_dict[idx] = call_stmt.output
                elif self.arg_dict[idx] != call_stmt.output:
                    return 0

            arg_idx = 0
            if call_stmt.arguments:
                for stmt_arg_idx, stmt_arg in enumerate(call_stmt.arguments):
                    if arg_idx < len(src_pattern.functionCall.arguments):
                        pattern_arg = src_pattern.functionCall.arguments[arg_idx]
                        arg_idx += 1
                        if isinstance(pattern_arg, Argument) and pattern_arg.value == '':
                            continue
                        if not pattern_arg.assigned:
                            idx = pattern_arg.variable[0]
                            if idx not in self.arg_dict:
                                return 0
                            elif not self.arg_dict[idx]:
                                self.arg_dict[idx] = stmt_arg
                            elif self.arg_dict[idx] != stmt_arg:
                                return 0
                    else:
                        kw_idx = arg_idx - len(src_pattern.functionCall.arguments)
                        if kw_idx < len(src_pattern.functionCall.keywords):
                            pattern_kw = src_pattern.functionCall.keywords[kw_idx]
                            pattern_kw_arg = pattern_kw.argument
                            if isinstance(pattern_kw_arg, Argument) and pattern_kw_arg.value == '':
                                continue
                            if not pattern_kw_arg.assigned:
                                idx = pattern_kw_arg.variable[0]
                                if idx not in self.arg_dict:
                                    return 0
                                elif not self.arg_dict[idx]:
                                    self.arg_dict[idx] = stmt_arg
                                elif self.arg_dict[idx] != stmt_arg:
                                    return 0

            if call_stmt.keywords:
                for stmt_kw_idx, stmt_kw in enumerate(call_stmt.keywords):
                    a=1
                    if stmt_kw not in [kw.key for kw in src_pattern.functionCall.keywords]:
                        continue
                    else:
                        idx = [kw.key for kw in src_pattern.functionCall.keywords].index(stmt_kw)
                        pattern_kw = src_pattern.functionCall.keywords[idx]
                        pattern_kw_arg = pattern_kw.argument
                        if isinstance(pattern_kw_arg, Argument) and pattern_kw_arg.value == '':
                            continue
                        if not pattern_kw_arg.assigned:
                            idx = pattern_kw_arg.variable[0]
                            if idx not in self.arg_dict:
                                return 0
                            elif not self.arg_dict[idx]:
                                self.arg_dict[idx] = call_stmt.keywords[stmt_kw]
                            elif self.arg_dict[idx] != call_stmt.keywords[stmt_kw]:
                                return 0

        if set([call_stmt.in_seq for call_stmt in call_sequence]) == {True}:
            a=1

        return 1

    def unorderedRefactor(self, call_sequence, idx_list, arg_dict):
        call_statements = [call_sequence.call_statements[idx] for idx in idx_list]
        lineno, tree_before, entry, idx = call_sequence.get_trees_from_index(idx_list)
        after = self._refactor(call_statements, arg_dict)
        if not after:
            return None, None
        for ii in idx:
            entry[ii] = None
        while None in entry:
            entry.remove(None)
        for i_tree_a, tree_a in enumerate(after):
            entry.insert(idx[0] + i_tree_a, gast.gast_to_ast(tree_a))
        return lineno, tree_before

    def refactor(self, call_sequence, i, j, arg_dict):
        lineno, tree_before, entry, idx = call_sequence.get_trees(i, j)
        after = self._refactor(call_sequence.call_statements[i: j], arg_dict)
        if not after:
            return None, None
        for ii in idx:
            entry[ii] = None
        while None in entry:
            entry.remove(None)
        for i_tree_a, tree_a in enumerate(after):
            entry.insert(idx[0] + i_tree_a, gast.gast_to_ast(tree_a))
        return lineno, tree_before

    def _refactor(self, call_stmts, arg_dict):
        temp_variable_dict = call_stmts[0].global_visitor.temp_var_dict
        if self.detect_only:
            return None
        package_mngr = call_stmts[0].global_visitor.package_mngr
        trees = []
        seq_tree = None
        insert_idx = None
        if set([call_stmt.in_seq for call_stmt in call_stmts]) == {True}:
            old_tree = call_stmts[0].seq_assign
            new_tree = copy.deepcopy(call_stmts[0].seq_assign)
            new_tree.value.args = []
            old_calls = [(call_stmt.ast_node.lineno, call_stmt.ast_node.col_offset) for call_stmt in call_stmts]
            insert_idx = len(old_tree.value.args)
            for idx, arg in enumerate(old_tree.value.args):
                if (arg.lineno, arg.col_offset) in old_calls:
                    if idx < insert_idx:
                        insert_idx = idx
                else:
                    new_tree.value.args.append(arg)
            seq_tree = new_tree

        for dest_pattern in self.destPatterns:
            if seq_tree:
                function_call = dest_pattern.functionCall.definition
            else:
                function_call = dest_pattern.functionCall
            tree = None
            if dest_pattern.fixed:
                tree = ast.parse(dest_pattern.fixedValue).body[0]
            else:
                tree = ast.parse('a=foo(x, y=z)').body[0]
                if isinstance(dest_pattern.output, list):
                    tree1 = ast.parse('a, b=foo(x, y=z)').body[0]
                    tuple = tree1.targets[0]
                    for idx, o in enumerate(dest_pattern.output):
                        target = copy.deepcopy(tuple.elts[0])
                        if o.assigned:
                            target.id = o.value
                        else:
                            _value = o.value
                            vars = re.findall(r'\$[a-zA-Z_][a-zA-Z0-9_]*', _value)
                            for var in vars:
                                _value = _value.replace(var, '.'.join(arg_dict[var]))
                            target.id = _value
                        if idx == 0 or idx == 1:
                            tuple.elts[idx] = target
                        else:
                            tuple.elts.append(target)
                    tree.targets[0] = copy.deepcopy(tuple)
                    del tree1
                elif not seq_tree:
                    if dest_pattern.output.assigned:
                        tree.targets[0].id = dest_pattern.output.value
                    else:
                        try:
                            _value = dest_pattern.output.value
                            vars = re.findall(r'\$[a-zA-Z_][a-zA-Z0-9_]*', _value)
                            for var in vars:
                                _value = _value.replace(var, '.'.join(arg_dict[var]))
                            tree.targets[0].id = _value
                        except:
                            continue

                if function_call.functionId.is_variable:
                    val = arg_dict[function_call.functionId.id]
                    function_id = '.'.join(val)
                else:
                    function_id = function_call.functionId.id
                if len(function_id.split('.')) > 1:
                    pkg = '.'.join(function_id.split('.')[:-1])
                else:
                    pkg = function_id
                if "self" not in pkg and pkg not in package_mngr.modules.values():
                    import_tree = ast.parse('from a import b as c').body[0]
                    import_tree.module = '.'.join(function_id.split('.')[:-1])
                    import_tree.names[0].name = function_id.split('.')[-1]
                    import_tree.names[0].asname = function_id.split('.')[-2] + '_' + function_id.split('.')[-1]
                    trees.append(import_tree)
                    tree.value.func.id = import_tree.names[0].asname
                else:
                    tree.value.func.id = function_id

                argument = copy.deepcopy(tree.value.args[0])
                tree.value.args = []
                for arg in function_call.arguments:
                    if arg.assigned:
                        tmp_argument = copy.deepcopy(argument)
                        tmp_argument.id = arg.value
                        tree.value.args.append(tmp_argument)
                    else:
                        tmp_argument = copy.deepcopy(argument)
                        nested = False
                        null = False
                        _value = arg.value
                        vars = re.findall(r'\$[a-zA-Z_][a-zA-Z0-9_]*', _value)
                        for var in vars:
                            if not arg_dict[var]:
                                null = True
                                continue
                            _value = _value.replace(var, '.'.join(arg_dict[var]))
                            if '@@temp' in arg_dict[var][0]:
                                n = temp_variable_dict[arg_dict[var][0]]
                                tmp_argument = n.ast_node
                                nested = True
                        if null:
                            continue
                        if not nested:
                            tmp_argument.id = _value
                        tree.value.args.append(tmp_argument)

                keyword = copy.deepcopy(tree.value.keywords[0])
                tree.value.keywords = []
                for kw in function_call.keywords:
                    tmp_keyword = copy.deepcopy(keyword)
                    tmp_keyword.arg = kw.key
                    if kw.argument.assigned:
                        tmp_keyword.value.id = kw.argument.value
                        tree.value.keywords.append(tmp_keyword)
                    else:
                        nested = False
                        null = False
                        _value = kw.argument.value
                        vars = re.findall(r'\$[a-zA-Z_][a-zA-Z0-9_]*', _value)
                        for var in vars:
                            if not arg_dict[var]:
                                null = True
                                continue
                            _value = _value.replace(var, '.'.join(arg_dict[var]))
                            if '@@temp' in arg_dict[var][0]:
                                n = temp_variable_dict[arg_dict[var][0]]
                                tmp_keyword.value = n.ast_node
                                nested = True
                        if null:
                            continue
                        if not nested:
                            tmp_keyword.value.id = _value
                        tree.value.keywords.append(tmp_keyword)
            try:
                if '@@temp' in tree.targets[0].id:
                    n = temp_variable_dict[tree.targets[0].id]
                    if n.p_entry == 'node.func.value':
                        n.p.ast_node.value.func.value = tree.value
                    tree = n.p.ast_node
            except:
                pass
            if seq_tree:
                seq_tree.value.args.insert(insert_idx, tree.value)
            else:
                trees.append(tree)
        if seq_tree:
            trees.append(seq_tree)
        return trees


class Assignment(object):
    def __init__(self, assignment, rule):
        self.fixed = False
        self.fixedValue = None
        self.output = ''
        self.functionCall = None
        self.Rule = rule

        if assignment.startswith('#'):
            self.fixed = True
            self.fixedValue = assignment[1:]
            return
        function_call, output = assignment.split('->')
        if len(output.split(',')) > 1:
            output = output.split(',')
        try:
            self.output = Argument(output.strip(), self.Rule)
        except AttributeError:
            self.output = [Argument(o.strip(), self.Rule) for o in output]
        self.functionCall = FunctionCall(function_call.strip(), self.Rule)


class FunctionCall(object):
    def __init__(self, func_call, rule):
        self.functionId = None
        self.arguments = []
        self.keywords = []
        self.definition = None
        self.Rule = rule

        parse = re.split(r'[(,)]', func_call)
        if '+' in parse[0]:
            self.functionId = function_Id('Add')
            self.arguments.append(Argument(parse[0].split('+')[0].strip(), self.Rule))
            self.arguments.append(Argument(parse[0].split('+')[1].strip(), self.Rule))
            return

        self.functionId = function_Id(parse[0])
        if len(parse) == 1:
            return
        for arg in parse[1:-1]:
            if '=' in arg:
                self.keywords.append(Keyword(arg.strip(), self.Rule))
            else:
                self.arguments.append(Argument(arg.strip(), self.Rule))


class Keyword(object):
    def __init__(self, argument, rule):
        self.key = ''
        self.argument = None
        self.Rule = rule

        key, arg = argument.split('=')
        self.key = key.strip()
        self.argument = Argument(arg.strip(), self.Rule)


class Argument(object):
    def __init__(self, argument, rule):
        self.assigned = False
        self.variable = None
        self.value = None
        self.Rule = rule

        # check if there is a substring starting with '$' using regex
        self.variable = re.findall(r'\$[a-zA-Z_][a-zA-Z0-9_]*', argument)
        if self.variable:
            self.assigned = False
        else:
            self.assigned = True

        if not self.assigned:
            for arg in self.variable:
                self.Rule.setArg(arg, None)
        self.value = argument


class function_Id(object):
    def __init__(self, func_id):
        if func_id.startswith('@'):
            self.id = func_id[1:]
            self.in_order = False
        else:
            self.id = func_id
            self.in_order = True
        self.is_variable = False


def main():
    pass


if __name__ == '__main__':
    main()
