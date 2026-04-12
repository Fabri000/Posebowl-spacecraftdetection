import torch

class Evaluator:
    def __init__(self, evaluation_fn):
        self.evaluation_fn = evaluation_fn

    def evaluate(self, model:torch.nn.Module, dataloader:torch.utils.data.DataLoader, device:torch.device):
        model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, targets in dataloader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                predictions = model(images)
                all_predictions.extend(predictions)
                all_targets.extend(targets)

        evaluation_results = self.evaluation_fn(all_predictions, all_targets)
        return evaluation_results