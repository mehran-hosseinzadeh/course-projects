"""
this class is implemented for you. you don't need to change anything!
"""


class Module:
    def __init__(self, name):
        """
        name: for each module you have to input a unique name.
        cache: you can use cache to save things you may need.
        phase: phase shows the module is in test or train phases.
        """
        self.name = name
        self.cache = None
        self.phase = 'Train'

    def test(self):
        self.phase = 'Test'

    def train(self):
        self.phase = 'Train'

    def forward(self, x, **kwargs):
        pass

    def backward(self, dout):
        pass
