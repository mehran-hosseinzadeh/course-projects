"""
this class is implemented for you. you don't need to change anything!
"""


import numpy as np


class Model:
    def __init__(self, optimizer):
        """
        optimizer: object of optimizer to update model.
        """
        self.modules = []
        self.optimizer = optimizer
    """
    train and test functions change phases of all modules in the model.
    """
    def train(self):
        for m in self.modules:
            m.train()

    def test(self):
        for m in self.modules:
            m.test()

    """
    fit function trains the model on X_train and y_train for "epochs" epochs.
    """
    def fit(self, X_train, y_train, X_valid, y_valid, batch_size=32, epochs=50):
        batch_count = int(len(X_train) / batch_size)
        losses = []
        accs = []
        val_losses = []
        val_accs = []

        for e in range(epochs):
            print('Epoch ' + str(e + 1), end=': ')
            batch_losses = []
            batch_accs = []
            for __ in range(batch_count):
                batch = np.random.choice(len(X_train), batch_size)
                batch_X, batch_y = X_train[batch], y_train[batch]
                current_X = batch_X
                for module in self.modules[:-1]:
                    current_X = module.forward(current_X)
                loss, probs = self.modules[-1].forward(current_X, y=batch_y)

                batch_accs.append(self.get_accuracy(probs, batch_y))
                batch_losses.append(loss)

                dout = self.modules[-1].backward(.0)
                for module in reversed(self.modules[:-1]):
                    dout = module.backward(dout)
                    self.optimizer.update(module)
            loss = np.array(batch_losses).mean()
            acc = np.array(batch_accs).mean()
            losses.append(loss)
            accs.append(acc)

            val_loss, val_acc = self.evaluate(X_valid, y_valid, batch_size)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            print('loss = {0:.4f}'.format(loss) + ', acc = {0:.4f}'.format(acc)
                  + ', val_loss = {0:.4f}'.format(val_loss) + ', val_acc = {0:.4f}'.format(val_acc))
            self.optimizer.next_iteration()
        return losses, accs, val_losses, val_accs

    def get_accuracy(self, probs, y):
        yhat = np.argmax(probs, axis=1)
        return np.sum(yhat == y) / len(y)

    def evaluate(self, X, y, batch_size):
        self.test()
        batch_count = int(len(X) / batch_size)
        batch_losses = []
        batch_accs = []
        for __ in range(batch_count):
            batch = np.random.choice(len(X), batch_size)
            batch_X, batch_y = X[batch], y[batch]
            current_X = batch_X
            for layer in self.modules[:-1]:
                current_X = layer.forward(current_X)
            loss, probs = self.modules[-1].forward(current_X, y=batch_y)
            batch_accs.append(self.get_accuracy(probs, batch_y))
            batch_losses.append(loss)
        loss = np.array(batch_losses).mean()
        acc = np.array(batch_accs).mean()
        self.train()
        return loss, acc

    def add(self, layer):
        self.modules.append(layer)
