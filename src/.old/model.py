from torch import nn
from transformers import DetrForObjectDetection
def get_model(model_name):
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if model_name == 'facebook/detr-resnet-50':
        return DetrModel()
        
    
class DetrModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        self.model.class_labels_classifier = torch.nn.Linear(in_features=256, out_features=15, bias=True)
        self.name = 'facebook/detr-resnet-50'

    def forward(self, x):
        x = self.model(x)
        return x
