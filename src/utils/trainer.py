import tqdm
import torch

class Trainer:
    def __init__(self, model:torch.nn.Module, optimizer:torch.optim.Optimizer, lr_scheduler:torch.optim.lr_scheduler._LRScheduler, device:torch.device):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device

    def train(self, train_dataloader:torch.utils.data.DataLoader, eval_dataloader:torch.utils.data.DataLoader, epochs:int = 1):
        results = {"train": [], "eval": []}

        for epoch in range(epochs):
            # --- Training Phase ---
            self.model.train()
            train_running_total_loss = 0.0
            
            pbar_train = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch+1} [Train]")
            for images, targets in pbar_train:
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

                train_running_total_loss += losses.item()
                pbar_train.set_postfix(loss=losses.item())

            self.lr_scheduler.step()
            avg_train_loss = train_running_total_loss / len(train_dataloader)

            eval_running_total_loss = 0.0
            pbar_eval = tqdm.tqdm(eval_dataloader, desc=f"Epoch {epoch+1} [Eval]")
            with torch.no_grad():
                for images, targets in pbar_eval:
                    images = list(image.to(self.device) for image in images)
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    
                    eval_running_total_loss += losses.item()
                    pbar_eval.set_postfix(loss=losses.item())

            avg_eval_loss = eval_running_total_loss / len(eval_dataloader)
            
            results["train"].append(avg_train_loss)
            results["eval"].append(avg_eval_loss)

        return results
    