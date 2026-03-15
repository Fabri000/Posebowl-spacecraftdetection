import tqdm
import torch

class Trainer:
    def __init__(self, model:torch.nn.Module, optimizer:torch.optim.Optimizer, lr_scheduler:torch.optim.lr_scheduler._LRScheduler, device:torch.device):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device

    def train(self, train_dataloader:torch.utils.data.DataLoader, eval_dataloader:torch.utils.data.DataLoader, epochs:int = 1):
        
        for _ in tqdm.tqdm(range(epochs)):
            train_loss = 0.0
            self.model.train()
            for images, targets in train_dataloader:
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                train_loss += losses.item()

                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

                self.lr_scheduler.step()
            
            eval_loss = 0.0
            self.model.eval()
            for images, targets in eval_dataloader:
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                eval_loss += losses.item()

            print(f"Epoch: {_}, Train Loss: {train_loss / len(train_dataloader)}, Eval Loss: {eval_loss / len(eval_dataloader)}")

        return 
    
    