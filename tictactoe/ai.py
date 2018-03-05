import random
import math
import random
import unittest
import networkx as nx
from typing import List

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from tictactoe import TicTacToe


class DepthFirstSearchAI(TicTacToe.Player):

    def __init__(self, *args, **kwargs):
        super(DepthFirstSearchAI, self).__init__(*args, **kwargs)
        self.graph = nx.DiGraph()
        self.best_moves = None

    def play(self) -> TicTacToe.Tile:
        tree = self.build_tree()
        nx.nx_pydot.write_dot(tree, 'tree.dot')
        nx.write_gexf(tree, 'tree.gexf')
        nx.write_graphml(tree, 'tree.graphml')
        pos = nx.nx_pydot.graphviz_layout(tree, prog='dot')
        nx.draw(tree, pos=pos, with_labels=True, arrows=True)
        plt.show()
        # TODO Extract the best move
        return random.choice(self.tictactoe.choices())

    def build_tree(self) -> nx.DiGraph:
        def expand_tree(tictactoe: TicTacToe, tree: nx.DiGraph, recursion_depth: int=1) -> nx.DiGraph:
            parent, child = str(tictactoe), tictactoe

            if recursion_depth > 2:
                return

            for tile in child.choices():
                child.set(tile)
                score = child.score(tile)
                if score is None:
                    tree.add_node(str(child), recursion_depth=recursion_depth)
                    tree.add_edge(parent, str(child))
                    expand_tree(child, tree, recursion_depth + 1)
                else:
                    tree.add_node(str(child), score=score, recursion_depth=recursion_depth)
                    tree.add_edge(parent, str(child))
                child.unset(tile)

        tree = nx.DiGraph(name='DFSTree')
        expand_tree(self.tictactoe, tree)
        return tree

    def reset(self):
        pass

    def visualize(self) -> bytes:
        pass


class MonteCarloTreeSearchAI(TicTacToe.Player):

    def __init__(self, *args, **kwargs):
        super(MonteCarloTreeSearchAI, self).__init__(*args, **kwargs)
        self.graph = nx.DiGraph()

    def play(self) -> TicTacToe.Tile:
        tictactoe = self.tictactoe

        def ulp_score(node, succ_node):
            node, succ_node = self.graph.node[node], self.graph.node[succ_node]
            return succ_node['num_wins'] / succ_node['num_visits'] + \
                   1.0 * math.sqrt(math.log(node['num_visits']) / succ_node['num_visits'])

        def select(node):
            if self.graph.successors(node):
                succ_ulp_scores = [(succ_node, ulp_score(node, succ_node)) for succ_node in self.graph.successors(node)]
                succ_node = max(succ_ulp_scores, key=lambda tpl: tpl[1])[0]
                tictactoe.set(self.graph.edge[node][succ_node]['move'])
                return select(succ_node)
            return node

        def expand(node):
            if self.graph.node[node]['score'] is None:
                for move in tictactoe.choices():
                    tictactoe.set(move)
                    succ_node, score = str(tictactoe), tictactoe.score(move)
                    self.graph.add_node(succ_node, attr_dict={'score': score, 'num_visits': 1, 'num_wins': 0})
                    self.graph.add_edge(node, succ_node, attr_dict={'move': move})
                    tictactoe.clear(move)
                playout_move = random.choice(tictactoe.choices())
                tictactoe.set(playout_move)
                score = tictactoe.score(playout_move)
                if score is None:
                    score = playout()
                return score
            return self.graph.node[node]['score']

        def playout():
            playout_move = random.choice(tictactoe.choices())
            tictactoe.set(playout_move)
            score = tictactoe.score(playout_move)
            if score is None:
                score = playout()
            tictactoe.clear(playout_move)
            return score

        def backpropagate(node, score):
            self.graph.node[node]['num_visits'] += 1
            self.graph.node[node]['num_wins'] += score
            if self.graph.predecessors(node):
                pred_node = self.graph.predecessors(node)[0]
                tictactoe.clear(self.graph.edge[pred_node][node]['move'])
                backpropagate(pred_node, score)

        repeat = 100
        if str(tictactoe) not in self.graph:
            self.graph.add_node(str(tictactoe), attr_dict={'score': None, 'num_visits': 0, 'num_wins': 0})
        root_node = str(tictactoe)

        while repeat > 0:
            selected_node = select(root_node)
            score = expand(selected_node)
            backpropagate(str(tictactoe), score)
            repeat -= 1

        succ_visits = [(succ_node, self.graph.node[succ_node]['num_visits']) for succ_node in self.graph.successors(root_node)]
        succ_node = max(succ_visits, key=lambda tpl: tpl[1])[0]
        move = self.graph.edge[root_node][succ_node]['move']
        return tictactoe[move]

    def visualize(self):
        position = nx.nx_agraph.graphviz_layout(self.graph, prog='dot')
        nx.draw(self.graph, position, with_labels=True, font_weight='bold')
        plt.show()

    def reset(self):
        self.graph.clear()


# region Unit Tests


class TestDepthFirstSearchAI(unittest.TestCase):

    Situations = {
        'Finish': [
            '#OX',
            'OXX',
            'OXO'],
        'EasyWin': [
            '#X-',
            'XOO',
            'XOO'],
        'DontScrewUp': [
            'OX-',
            'OX-',
            '#OX'],
        'DontMessUp1': [
            '#-X',
            'OX-',
            'OXO'],
        'DontMessUp2': [
            '#-X',
            'O--',
            'OX-'],
        'DontF__kUp': [
            '-#-',
            '-O-',
            '-OX']
    }

    @staticmethod
    def find(scenario: List[str], char: str) -> tuple:
        row_line_with_char = [(row, line) for row, line in enumerate(scenario) if char in line]
        assert len(row_line_with_char) == 1
        row, line = row_line_with_char[0]
        return row, line.find(char)

    def play(self, scenario: List[str], o: TicTacToe.Player, x: TicTacToe.Player):
        tictactoe = TicTacToe.build(scenario, o=o, x=x)
        tile = x.play()
        correct = self.find(scenario, '#')
        self.assertEqual((tile.row, tile.column), correct)

    def test_basics(self):
        dummy = TicTacToe.Player('O')
        ai = DepthFirstSearchAI('X')
        self.play(self.Situations['Finish'], o=dummy, x=ai)
        self.play(self.Situations['EasyWin'], o=dummy, x=ai)
        self.play(self.Situations['DontScrewUp'], o=dummy, x=ai)
        self.play(self.Situations['DontMessUp1'], o=dummy, x=ai)
        self.play(self.Situations['DontMessUp2'], o=dummy, x=ai)
        self.play(self.Situations['DontF__kUp'], o=dummy, x=ai)

    def test_ai_vs_ai(self):
        o, x = DepthFirstSearchAI('O'), DepthFirstSearchAI('X')
        tictactoe = TicTacToe(o, x)
        while True:
            score = tictactoe.round()
            if score is not None:
                break
        self.assertEqual(score, 0, "AI vs AI game must always end up in a tie:\n" + str(tictactoe))


class TestMonteCarloSearchAI(TestDepthFirstSearchAI):

    def test_basics(self):
        dummy = TicTacToe.Player('O')
        ai = MonteCarloTreeSearchAI('X')
        self.play(self.Situations['Finish'], o=dummy, x=ai)
        self.play(self.Situations['EasyWin'], o=dummy, x=ai)
        self.play(self.Situations['DontScrewUp'], o=dummy, x=ai)
        self.play(self.Situations['DontMessUp1'], o=dummy, x=ai)
        self.play(self.Situations['DontMessUp2'], o=dummy, x=ai)
        self.play(self.Situations['DontF__kUp'], o=dummy, x=ai)

    def test_ai_vs_ai(self):
        raise NotImplementedError()


# endregion
