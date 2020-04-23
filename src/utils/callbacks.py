class EarlyStopping:
    def __init__(self, patience=0):
        self.last_metrics = 10**8
        self.patience = patience
        self.patience_count = 0

    def check_training(self, metric):
        if metric < self.last_metrics:
            self.last_metrics = metric
            self.patience_count = 0
            return False
        elif (metric > self.last_metrics) & (self.patience_count < self.patience):
            self.patience_count += 1
            return False
        else:
            return True

