
import torch
from transformers import DetrConfig, DetrForObjectDetection, Detr
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
        
        DETR resizes the input images such that the shortest side is at least a certain amount of pixels while the longest 
        is at most 1333 pixels. At training time, scale augmentation is used such that the shortest side is randomly set to 
        at least 480 and at most 800 pixels. At inference time, the shortest side is set to 800. One can use 
        DetrImageProcessor to prepare images (and optional annotations in COCO format) for the model. Due to this resizing, 
        images in a batch can have different sizes. DETR solves this by padding images up to the largest size in a batch, 
        and by creating a pixel mask that indicates which pixels are real/which are padding. Alternatively, one can also 
        define a custom collate_fn in order to batch images together, using ~transformers.DetrImageProcessor.pad_and_create_pixel_mask.
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

