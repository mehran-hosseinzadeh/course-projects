"""
this class is implemented for you. you don't need to change anything!
"""


class Optimizer:
    def __init__(self, learning_rate=1e-3):
        self.learning_rate = learning_rate
        self.iteration_number = 1

    def update(self, layer):
        pass

    def next_iteration(self):
        self.iteration_number += 1
