import numpy as np


class DraftNode:
    def __init__(
            self, blue_bans, red_bans,
            blue_picks, red_picks, is_max,
            evaluation_function,
            alpha=-np.inf, beta=np.inf):
        self.blue_bans = blue_bans
        self.red_bans = red_bans
        self.blue_picks = blue_picks
        self.red_picks = red_picks
        self.is_max = is_max
        self.children = []
        self.evaluate = evaluation_function
        self.alpha = alpha
        self.beta = beta

    def is_terminal(self):
        return len(self.blue_picks) == 5 or len(self.red_picks) == 5


"""
Order of picks/bans
0: ban
1: pick
"""
__action_order = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
"""
Order of team doing the action
0: blue
1: red
"""
__team_order = [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1]


def minimax_for_blue(node, depth, order_index, alpha, beta):
    """
    Calculates the best next action for blue to take on a given draft round
    :param node: search space node to evaluate
    :param depth: depth of the current node
    :param order_index: index of the draft order
    :param alpha:
    :param beta:
    :return:
    """
    if depth == 0 or node.is_terminal():
        return node.evaluate()

    current_action = __action_order[order_index]
    current_team = __team_order[order_index]
    order_index += 1

    if current_action == 0:
        # Ban
        if current_team == 0:
            # Blue ban
            max_eval = -np.inf
            for child in node.children:
                eval = minimax_for_blue(child, depth - 1, order_index, alpha, beta)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if alpha >= beta:
                    break  # Prune the rest of the subtree
            return max_eval
        else:
            # Red ban
            min_eval = np.inf
            for child in node.children:
                eval = minimax_for_blue(child, depth - 1, order_index, alpha, beta)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if alpha >= beta:
                    break  # Prune the rest of the subtree
            return min_eval
    else:
        # Pick
        if current_team == 0:
            # Blue pick
            max_eval = -np.inf
            for child in node.children:
                eval = minimax_for_blue(child, depth - 1, order_index, alpha, beta)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if alpha >= beta:
                    break  # Prune the rest of the subtree
            return max_eval
        else:
            # Red pick
            min_eval = np.inf
            for child in node.children:
                eval = minimax_for_blue(child, depth - 1, order_index, alpha, beta)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if alpha >= beta:
                    break  # Prune the rest of the subtree
            return min_eval
