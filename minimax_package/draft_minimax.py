import numpy as np


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


class DraftNode:
    def __init__(
            self, blue_bans, red_bans,
            blue_picks, red_picks, order_index,
            evaluation_function):
        self.blue_bans = blue_bans
        self.red_bans = red_bans
        self.blue_picks = blue_picks
        self.red_picks = red_picks
        self.order_index = order_index
        self.evaluate = evaluation_function

    def is_terminal(self):
        """
        Determines whether this node is terminal
        :return: true/false
        """
        return self.order_index >= 20

    def get_children(self, next_action, next_team):
        """
        Returns a list of all possible child nodes.
        :param next_action: Action taken to create the child nodes [0: ban, 1: pick]
        :param next_team: Team that performed the action to create the child node [0: blue, 1: red]
        :return: list of all possible child nodes
        """
        if next_action == 0:
            # stub
            return self
        else:
            # stub
            return self


def iterative_deepening(node, max_depth, top_n=10):
    """
    Performs minimax to a defined depth, providing the best found actions.
    Requires that partial evaluation be implemented.
    :param node: root tree node
    :param max_depth: maximum search depth
    :param top_n: number of top actions to retain
    :return: list of the top n best actions in reverse sorted order
    """
    # Store top n best moves in reverse sorted order
    top_moves = []
    # Initialize alpha and beta
    alpha, beta = -np.inf, np.inf
    for depth in range(1, max_depth + 1):
        action, score = minimax_for_blue(node, depth, max_depth, 0, alpha, beta)
        if action is not None:
            # Attempt to add move to the list
            top_moves.append((action, score))
            # Sort the list by the score in reverse order
            top_moves.sort(key=lambda x: x[1], reverse=True)
            # Only retain the first 10 elements of the list
            top_moves = top_moves[:top_n]
    return top_moves


def minimax(maximize_blue, node, depth, max_depth, order_index, alpha, beta):
    """
    Calculates the best next action for blue to take on a given draft round
    :param maximize_blue: finding the best actions for blue team [True/False]
    :param node: search space node to evaluate
    :param depth: depth of the current node
    :param max_depth: depth of an early termination layer
    :param order_index: index of the draft order
    :param alpha: The best value found so far for the maximizing team
    :param beta: The best value found so far for the minimizing team
    :return:
    """
    best_action = None
    if depth == 0 or node.is_terminal():
        return node.evaluate()

    # Remove if partial evaluation is not finished
    if depth >= max_depth:
        return node.evaluate()

    current_team = __team_order[order_index]
    order_index += 1

    if maximize_blue:
        return minimax_for_blue(node, depth, max_depth, order_index, alpha, beta, current_team)
    else:
        return minimax_for_red(node, depth, max_depth, order_index, alpha, beta, current_team)


def minimax_for_blue(node, depth, max_depth, order_index, alpha, beta, current_team):
    """
    Calculates the best next action for blue to take on a given draft round
    :param node: search space node to evaluate
    :param depth: depth of the current node
    :param max_depth: depth of an early termination layer
    :param order_index: index of the draft order
    :param alpha: The best value found so far for the maximizing team
    :param beta: The best value found so far for the minimizing team
    :param current_team: value representing which team made the current action
    :return:
    """
    best_action = None
    next_action = __action_order[order_index]
    next_team = __team_order[order_index]

    if current_team == 0:
        # Blue
        max_eval = -np.inf
        for child in node.getChildren(next_action, next_team):
            _, eval = minimax(True, child, depth + 1, max_depth, order_index, alpha, beta)
            if eval > max_eval:
                max_eval = eval
                best_action = child  # Store the best action
            alpha = max(alpha, eval)
            if alpha >= beta:
                break  # Prune the rest of the subtree
            return best_action, max_eval
    else:
        # Red
        min_eval = np.inf
        for child in node.getChildren(next_action, next_team):
            _, eval = minimax(True, child, depth + 1, max_depth, order_index, alpha, beta)
            if eval < min_eval:
                min_eval = eval
                best_action = child  # Store the best action
            beta = min(beta, eval)
            if alpha >= beta:
                break  # Prune the rest of the subtree
        return best_action, min_eval


def minimax_for_red(node, depth, max_depth, order_index, alpha, beta, current_team):
    """
    Calculates the best next action for blue to take on a given draft round
    :param node: search space node to evaluate
    :param depth: depth of the current node
    :param max_depth: depth of an early termination layer
    :param order_index: index of the draft order
    :param alpha: The best value found so far for the maximizing team
    :param beta: The best value found so far for the minimizing team
    :param current_team: value representing which team made the current action
    :return:
    """
    best_action = None
    next_action = __action_order[order_index]
    next_team = __team_order[order_index]

    if current_team == 0:
        # Blue
        min_eval = np.inf
        for child in node.getChildren(next_action, next_team):
            _, eval = minimax(False, child, depth + 1, max_depth, order_index, alpha, beta)
            if eval < min_eval:
                min_eval = eval
                best_action = child  # Store the best action
            beta = min(beta, eval)
            if alpha >= beta:
                break  # Prune the rest of the subtree
        return best_action, min_eval
    else:
        # Red
        max_eval = -np.inf
        for child in node.getChildren(next_action, next_team):
            _, eval = minimax(False, child, depth + 1, max_depth, order_index, alpha, beta)
            if eval > max_eval:
                max_eval = eval
                best_action = child  # Store the best action
            alpha = max(alpha, eval)
            if alpha >= beta:
                break  # Prune the rest of the subtree
        return best_action, max_eval
