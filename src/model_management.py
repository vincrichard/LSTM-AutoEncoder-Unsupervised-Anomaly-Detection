import torch


class ModelManagement:
    def __init__(self, path, name_model):
        self.path = path
        self.last_metrics = 10**8
        self.name_model = name_model

    def save(self, model):
        torch.save(model.state_dict(), self.path + '%s' % self.name_model)

    def checkpoint(self, epoch, model, optimizer, loss):
        if self.last_metrics > loss:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, self.path+'%s_%d' %(self.name_model,epoch))

