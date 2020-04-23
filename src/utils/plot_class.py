import matplotlib.pyplot as plt


class LossCheckpoint:
    def __init__(self):
        self.losses = []

    def plot(self, log=False):
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(self.losses)), self.losses)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        if log:
            plt.yscale('log')
        plt.show()
