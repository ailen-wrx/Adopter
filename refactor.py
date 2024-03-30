import ast, astunparse

import gast

RULES = ['nn.Linear@x#y->nn.ReLU#x=>fused_fc_relu@x#y', 'nn.LayerNorm=>apex.LayerNorm']


class moduleRefactor(object):
    def __init__(self, _module):
        self.module = _module

    def refactor(self, target_nodes, patterns, dests):
        pattern_funcs = [patterns[i]['func'].split('.') for i in range(len(patterns))]
        dest_funcs = [dests[i]['func'].split('.') for i in range(len(dests))]

        args = {}
        targets = {}

        for i in range(len(target_nodes)):
            node = target_nodes[i]

            astn = gast.gast_to_ast(node)

            for idx, symbol in enumerate(patterns[i]['args']):
                if symbol != '':
                    args[symbol] = node.value.args[idx]
            targets[patterns[i]['target']] = node.targets

        for i, dest in enumerate(dests):
            func = dest['func']
            call_node = ast.parse(f'a = {func}()').body[0]
            for j in dest['args']:
                call_node.value.args.append(args[j])
            call_node.targets = targets[dest['target']]

            print(astunparse.unparse(call_node))


        pass

