
import torch
from transformers import DetrConfig, DetrForObjectDetection
from ultralytics import YOLO


class HuggingfaceModel:
    def __init__(self):
        config = DetrConfig()
        # self.model = DetrForObjectDetection(config)
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")


    def forward(self, images: torch.Tensor, targets: list[dict] = None):
        """
        pixel_values [B, C, H, W] Normalize IageNetv1 mean std -~+
        labels (`List[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
            respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
            in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.

        """
        labels = {}
        labels["boxes"] = targets["boxes"]
        labels["class_labels"] = targets["class_labels"]
        labels = [{k: v.cuda()} for label in labels for k, v in label.items()]


        outputs = self.model(images, labels=labels)
        loss = outputs.loss 
        predicted_boxes = outputs.pred_boxes
        outputs = outputs.logits

        return_dict = {}
        return_dict["loss"] = loss
        return_dict["outputs"] = outputs    
        return_dict["predicted_boxes"] = predicted_boxes

        return return_dict


# class UltralyticsModel:
#     def __init__(self):
#         self.model = YOLO("yolov10s.pt")

#     def forward(self, images: torch.Tensor, targets: list[dict] = None):
#         self.model(images)

