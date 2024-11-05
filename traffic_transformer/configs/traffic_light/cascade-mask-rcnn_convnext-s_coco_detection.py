_base_ = '../convnext/cascade-mask-rcnn_convnext-t-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py'  # noqa

# please install mmpretrain
# import mmpretrain.models to trigger register_module in mmpretrain
dataset_type = 'CocoDataset'
data_root = '/workspace/traffic_light/data/detection/'
metainfo = {
    'classes' : (
        "veh_go",
        "veh_goLeft",
        "veh_noSign",
        "veh_stop",
        "veh_stopLeft",
        "veh_stopWarning",
        "veh_warning",
        "ped_go",
        "ped_noSign",
        "ped_stop",
        "bus_go",
        "bus_noSign",
        "bus_stop",
        "bus_warning",
    )
}

custom_imports = dict(
    imports=['mmpretrain.models'], allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-small_3rdparty_32xb128-noema_in1k_20220301-303e75e3.pth'  # noqa

train_dataloader = dict(
    batch_size=10,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/train.json',
        data_prefix=dict(img='train/images/'),
        ))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val/val.json',
        data_prefix=dict(img='val/images/'),))

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs')]

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test/test.json',
        data_prefix=dict(img='test/images/'),
        test_mode=True,
        pipeline=test_pipeline))

val_evaluator = dict(
    ann_file=data_root + 'val/val.json',)

test_evaluator = dict(
    type='CocoMetric',
    metric='bbox',
    format_only=True,
    ann_file=data_root + 'test/test.json',
    outfile_prefix='./work_dirs/traffic_detection/test')

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmpretrain.ConvNeXt',
        arch='small',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.6,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')))



optim_wrapper = dict(paramwise_cfg={
    'decay_rate': 0.7,
    'decay_type': 'layer_wise',
    'num_layers': 12
})
