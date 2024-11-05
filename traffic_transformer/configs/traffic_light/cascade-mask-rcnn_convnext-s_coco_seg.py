_base_ = '../convnext/cascade-mask-rcnn_convnext-t-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py'  # noqa

# please install mmpretrain
# import mmpretrain.models to trigger register_module in mmpretrain
dataset_type = 'CocoInsDataset'
data_root = '/workspace/traffic_light/data/segmentation_coco/'
metainfo = {
    'classes': (
        'Car_VehLane',
        'Car_VehLane_IncatLft',
        'Car_VehLane_IncatRht',
        'Car_VehLane_HazLit',
        'Car_VehLane_Brake',
        'Car_VehLane_Brake_IncatLft',
        'Car_VehLane_Brake_IncatRht',
        'Car_VehLane_Brake_HazLit',
        'Car_OutgoLane',
        'Car_OutgoLane_IncatLft',
        'Car_OutgoLane_IncatRht',
        'Car_OutgoLane_HazLit',
        'Car_OutgoLane_Brake',
        'Car_OutgoLane_Brake_IncatLft',
        'Car_OutgoLane_Brake_IncatRht',
        'Car_OutgoLane_Brake_HazLit',
        'Car_IncomLane',
        'Car_IncomLane_IncatLft',
        'Car_IncomLane_IncatRht',
        'Car_IncomLane_HazLit',
        'Car_IncomLane_Brake',
        'Car_IncomLane_Brake_IncatLft',
        'Car_IncomLane_Brake_IncatRht',
        'Car_IncomLane_Brake_HazLit',
        'Car_Jun',
        'Car_Jun_IncatLft',
        'Car_Jun_IncatRht',
        'Car_Jun_HazLit',
        'Car_Jun_Brake',
        'Car_Jun_Brake_IncatLft',
        'Car_Jun_Brake_IncatRht',
        'Car_Jun_Brake_HazLit',
        'Car_Parking',
        'Car_Parking_IncatLft',
        'Car_Parking_IncatRht',
        'Car_Parking_HazLit',
        'Car_Parking_Brake',
        'Car_Parking_Brake_IncatLft',
        'Car_Parking_Brake_IncatRht',
        'Car_Parking_Brake_HazLit',
        'Bus_VehLane',
        'Bus_VehLane_IncatLft',
        'Bus_VehLane_IncatRht',
        'Bus_VehLane_HazLit',
        'Bus_VehLane_Brake',
        'Bus_VehLane_Brake_IncatLft',
        'Bus_VehLane_Brake_IncatRht',
        'Bus_VehLane_Brake_HazLit',
        'Bus_OutgoLane',
        'Bus_OutgoLane_IncatLft',
        'Bus_OutgoLane_IncatRht',
        'Bus_OutgoLane_HazLit',
        'Bus_OutgoLane_Brake',
        'Bus_OutgoLane_Brake_IncatLft',
        'Bus_OutgoLane_Brake_IncatRht',
        'Bus_OutgoLane_Brake_HazLit',
        'Bus_IncomLane',
        'Bus_IncomLane_IncatLft',
        'Bus_IncomLane_IncatRht',
        'Bus_IncomLane_HazLit',
        'Bus_IncomLane_Brake',
        'Bus_IncomLane_Brake_IncatLft',
        'Bus_IncomLane_Brake_IncatRht',
        'Bus_IncomLane_Brake_HazLit',
        'Bus_Jun',
        'Bus_Jun_IncatLft',
        'Bus_Jun_IncatRht',
        'Bus_Jun_HazLit',
        'Bus_Jun_Brake',
        'Bus_Jun_Brake_IncatRht',
        'Bus_Jun_Brake_IncatLft',
        'Bus_Jun_Brake_HazLit',
        'Bus_Parking',
        'Bus_Parking_IncatLft',
        'Bus_Parking_IncatRht',
        'Bus_Parking_HazLit',
        'Bus_Parking_Brake',
        'Bus_Parking_Brake_IncatLft',
        'Bus_Parking_Brake_IncatRht',
        'Bus_Parking_Brake_HazLit'
    ),
    # palette is a list of color tuples, which is used for visualization.
    'palette':[
        (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
        (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
        (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
        (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
        (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
        (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
        (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
        (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
        (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
        (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
        (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
        (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
        (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
        (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
        (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
        (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
        (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
        (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
        (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
        (246, 0, 122), (191, 162, 208)
    ]}

custom_imports = dict(
    imports=['mmpretrain.models'], allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-small_3rdparty_32xb128-noema_in1k_20220301-303e75e3.pth'  # noqa

train_dataloader = dict(
    batch_size=12,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='labels/train2.json',
        data_prefix=dict(img='images/train2/'),
        ))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='labels/val2.json',
        data_prefix=dict(img='images/val2')
    ))

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
        ann_file='labels/test.json',
        data_prefix=dict(img='images/test'),
        test_mode=True,
        pipeline=test_pipeline))

val_evaluator = dict(
    ann_file=data_root + 'labels/val2.json',)

test_evaluator = dict(
    type='CocoMetric',
    format_only=True,
    ann_file=data_root + 'labels/test.json',
    outfile_prefix='./work_dirs/segmentation/test')

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
