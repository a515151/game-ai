import math
import random
import unittest
import networkx as nx
from typing import List, Tuple

from tictactoe import TicTacToe


class DepthFirstSearchAI(TicTacToe.Player):

    def play(self) -> TicTacToe.Tile:
        def best_move(tictactoe: TicTacToe, recursion_level: int=1) -> Tuple[int, TicTacToe.Tile]:
            best_score, best_tile = -math.inf, None

            for tile in tictactoe.choices():
                tictactoe.set(tile)
                score = tictactoe.score(tile)
                if score is None:
                    opponent_score, opponent_tile = best_move(tictactoe, recursion_level + 1)
                    score = -opponent_score
                else:
                    score /= recursion_level
                if score > best_score:
                    best_score, best_tile = score, tile
                tictactoe.unset(tile)
            return best_score, best_tile

        best_score, best_tile = best_move(self.tictactoe)
        return best_tile


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
