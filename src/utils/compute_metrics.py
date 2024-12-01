import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.nn.functional import softmax

batch_metrics = []
id2label={0: 'veh_go',
              1: 'veh_goLeft',
              2: 'veh_noSign',
              3: 'veh_stop',
              4: 'veh_stopLeft',
              5: 'veh_stopWarning',
              6: 'veh_warning',
              7: 'ped_go',
              8: 'ped_noSign',
              9: 'ped_stop',
              10: 'bus_go',
              11: 'bus_noSign',
              12: 'bus_stop',
              13: 'bus_warning'}

def denormalize_boxes(boxes, width, height):
    boxes = boxes.clone()
    boxes[:, 0] *= width  # xmin
    boxes[:, 1] *= height  # ymin
    boxes[:, 2] *= width  # xmax
    boxes[:, 3] *= height  # ymax
    return boxes

def compute_metrics(scores, pred_boxes, labels, compute_result):
    
    global batch_metrics, id2label

    image_sizes = []
    target = []
    for label in labels:

        image_sizes.append(label["orig_size"])
        width, height = label["orig_size"]
        denormalized_boxes = label["boxes"]#denormalize_boxes(label["boxes"], width, height)
        target.append(
            {
                "boxes": denormalized_boxes,
                "labels": label["class_labels"],
            }
        )
    predictions = []
    for score, box, target_sizes in zip(scores, pred_boxes, image_sizes):
        # Extract the bounding boxes, labels, and scores from the model's output
        pred_scores = score[:, :-1]  # Exclude the no-object class
        pred_scores = softmax(pred_scores, dim=-1)
        width, height = target_sizes
        pred_boxes = denormalize_boxes(box, width, height)
        pred_labels = torch.argmax(pred_scores, dim=-1)

        # Get the scores corresponding to the predicted labels
        pred_scores_for_labels = torch.gather(pred_scores, 1, pred_labels.unsqueeze(-1)).squeeze(-1)
        predictions.append(
            {
                "boxes": pred_boxes,
                "scores": pred_scores_for_labels,
                "labels": pred_labels,
            }
        )

    metric = MeanAveragePrecision(box_format="xywh", class_metrics=True)

    if not compute_result:
        # Accumulate batch-level metrics
        batch_metrics.append({"preds": predictions, "target": target})
        return {}
    else:
        # Compute final aggregated metrics
        # Aggregate batch-level metrics (this should be done based on your metric library's needs)
        all_preds = []
        all_targets = []
        for batch in batch_metrics:
            all_preds.extend(batch["preds"])
            all_targets.extend(batch["target"])

        # Update metric with all accumulated predictions and targets
        metric.update(preds=all_preds, target=all_targets)
        metrics = metric.compute()

        # Convert and format metrics as needed
        classes = metrics.pop("classes")
        map_per_class = metrics.pop("map_per_class")
        # mar_100_per_class = metrics.pop("mar_100_per_class")
        total_map=metrics.pop("map")
        map_50=metrics.pop("map_50")
        map_75=metrics.pop("map_75")
        for class_id, class_map in zip(classes, map_per_class):
            class_name = id2label[class_id.item()] if id2label is not None else class_id.item()
            metrics[f"mAP_{class_name}"] = class_map


        # Round metrics for cleaner output
        # metrics = {k: round(v.item(), 4) for k, v in metrics.items()}
        metrics['mAP']=total_map
        metrics['mAP_50']=map_50
        metrics['mAP_75']=map_75
        
        # Clear batch metrics for next evaluation
        batch_metrics = []

        return metrics