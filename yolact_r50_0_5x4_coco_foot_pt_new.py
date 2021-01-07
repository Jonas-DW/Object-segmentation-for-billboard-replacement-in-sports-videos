checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/home/student_gpu/mmdetection/checkpoints/yolact_r50_1x8_coco_20200908-f38d58df.pth'
resume_from = None
workflow = [('train', 1)]
img_size = 550
model = dict(
    type='YOLACT',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        zero_init_residual=False,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5,
        upsample_cfg=dict(mode='bilinear')),
    bbox_head=dict(
        type='YOLACTHead',
        num_classes=80,
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=3,
            scales_per_octave=1,
            base_sizes=[8, 16, 32, 64, 128],
            ratios=[0.5, 1.0, 2.0],
            strides=[
                7.971014492753623, 15.714285714285714, 30.555555555555557,
                61.111111111111114, 110.0
            ],
            centers=[(3.9855072463768115, 3.9855072463768115),
                     (7.857142857142857, 7.857142857142857),
                     (15.277777777777779, 15.277777777777779),
                     (30.555555555555557, 30.555555555555557), (55.0, 55.0)]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            reduction='none',
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.5),
        num_head_convs=1,
        num_protos=32,
        use_ohem=True),
    mask_head=dict(
        type='YOLACTProtonet',
        in_channels=256,
        num_protos=32,
        num_classes=80,
        max_masks_to_train=100,
        loss_mask_weight=6.125),
    segm_head=dict(
        type='YOLACTSegmHead',
        num_classes=80,
        in_channels=256,
        loss_segm=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0.0,
        ignore_iof_thr=-1,
        gt_max_assign_all=False),
    allowed_border=-1,
    pos_weight=-1,
    neg_pos_ratio=3,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    iou_thr=0.5,
    top_k=200,
    max_per_img=100)
dataset_type = 'CocoDataset'
classes = ('person', 'sports ball')
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.68, 116.78, 103.94], std=[58.4, 57.12, 57.38], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(4.0, 4.0)),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Expand',
        mean=[123.68, 116.78, 103.94],
        to_rgb=True,
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(550, 550), keep_ratio=False),
    dict(
        type='Normalize',
        mean=[123.68, 116.78, 103.94],
        std=[58.4, 57.12, 57.38],
        to_rgb=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(550, 550),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(
                type='Normalize',
                mean=[123.68, 116.78, 103.94],
                std=[58.4, 57.12, 57.38],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        classes=('person', 'sports ball'),
        ann_file='data/coco/annotations/cocofoot_train_full.json',
        img_prefix='data/coco/train2017/',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='FilterAnnotations', min_gt_bbox_wh=(4.0, 4.0)),
            dict(
                type='PhotoMetricDistortion',
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18),
            dict(
                type='Expand',
                mean=[123.68, 116.78, 103.94],
                to_rgb=True,
                ratio_range=(1, 4)),
            dict(
                type='MinIoURandomCrop',
                min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                min_crop_size=0.3),
            dict(type='Resize', img_scale=(550, 550), keep_ratio=False),
            dict(
                type='Normalize',
                mean=[123.68, 116.78, 103.94],
                std=[58.4, 57.12, 57.38],
                to_rgb=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
        ]),
    val=dict(
        type='CocoDataset',
        classes=('person', 'sports ball'),
        ann_file='data/custom/annotations/instances_coco_80.json',
        img_prefix='data/custom/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(550, 550),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(
                        type='Normalize',
                        mean=[123.68, 116.78, 103.94],
                        std=[58.4, 57.12, 57.38],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        classes=('person', 'sports ball'),
        ann_file='data/custom/annotations/instances_coco_80.json',
        img_prefix='data/custom/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(550, 550),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(
                        type='Normalize',
                        mean=[123.68, 116.78, 103.94],
                        std=[58.4, 57.12, 57.38],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(metric=['segm'])
optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    step=[20, 42, 49, 52])
total_epochs = 55
cudnn_benchmark = True
work_dir = './work_dirs/yolact_r50_0_5x4_coco_foot_pt_new'
gpu_ids = range(0, 1)
