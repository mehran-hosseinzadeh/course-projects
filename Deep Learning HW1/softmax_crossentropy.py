import numpy as np
from module import Module


class SoftmaxCrossentropy(Module):
    def __init__(self, name):
        super(SoftmaxCrossentropy, self).__init__(name)

    def forward(self, x, **kwargs):
        y = kwargs.pop('y', None)
        """
        x: input array.
        y: real labels for this input.
        probs: probabilities of labels for this input.
        loss: cross entropy loss between probs and real labels.
        **Save whatever you need for backward pass in self.cache.
        """
        # in order to prevent overflow, we subtract max of x_i from each input as proposed in Q4
        normalized_x = x - np.max(x, axis = 1).reshape(-1, 1)
        exp_values = np.exp(normalized_x)
        probs = exp_values / np.sum(exp_values, 1).reshape(-1, 1)

        loss = -np.log(probs[range(y.shape[0]), y])
        loss = np.sum(loss) / y.shape[0]
        self.cache = {'probs': probs, 'y': y}
        # todo: implement the forward propagation for probs and compute cross entropy loss
        # NOTE: implement a numerically stable version.If you are not careful here
        # it is easy to run into numeric instability!
   
        return loss, probs

    def backward(self, dout=0):
        dx = self.cache['probs']
        y = self.cache['y']
        dx[range(y.shape[0]), y] -= 1
        dx = dx / y.shape[0]

        return dx
