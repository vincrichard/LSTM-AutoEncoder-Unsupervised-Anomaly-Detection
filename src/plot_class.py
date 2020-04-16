import matplotlib.pyplot as plt


class PlotLoss:
    def __init__(self):
        self.losses = []

    def plot(self):
        plt.figure(figsize=(10,5))
        plt.plot(range(len(self.losses)), self.losses)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()
