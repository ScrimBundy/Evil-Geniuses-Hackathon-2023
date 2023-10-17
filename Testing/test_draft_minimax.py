from minimax_package.draft_minimax import DraftNode
import numpy as np


# Define your custom evaluation function here
def custom_evaluation(node):
    # Implement your evaluation logic
    return 0  # Replace with your evaluation score


def test():
    # Instantiate the DraftNode class with the custom evaluation function
    root = DraftNode([], [], [], [], True, custom_evaluation)


if __name__ == '__main__':
    test()
