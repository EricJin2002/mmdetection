custom_imports = dict(imports=['checkpoints.custom_transforms'], allow_failed_imports=False)
load_from = 'checkpoints/mask_rcnn_r101_fpn_mstrain-poly_3x_coco_20210524_200244-5675c317.pth'

metainfo = {
    "classes": (
        'AlarmClock', 'Apple', 'ArmChair', 'BaseballBat', 'BasketBall', 'Bed', 'Book', 'Boots', 'Bottle', 'Bowl', 'Box', 'Bread', 'ButterKnife', 'CD', 'Candle', 'Cart', 'CellPhone', 'Chair', 'Cloth', 'ClothesDryer', 'CoffeeMachine', 'CoffeeTable', 'CounterTop', 'CreditCard', 'Cup', 'Desk', 'DeskLamp', 'Desktop', 'DiningTable', 'DishSponge', 'DogBed', 'Dresser', 'Dumbbell', 'Egg', 'Faucet', 'FloorLamp', 'Footstool', 'Fork', 'Fridge', 'GarbageBag', 'GarbageCan', 'HandTowel', 'HandTowelHolder', 'HousePlant', 'Kettle', 'KeyChain', 'Knife', 'Ladle', 'Laptop', 'LaundryHamper', 'Lettuce', 'LightSwitch', 'Microwave', 'Mug', 'Newspaper', 'Ottoman', 'Painting', 'Pan', 'PaperTowelRoll', 'Pen', 'Pencil', 'PepperShaker', 'Pillow', 'Plate', 'Plunger', 'Pot', 'Potato', 'RemoteControl', 'Safe', 'SaltShaker', 'ScrubBrush', 'ShelvingUnit', 'ShowerCurtain', 'ShowerHead', 'SideTable', 'Sink', 'SoapBar', 'SoapBottle', 'Sofa', 'Spatula', 'Spoon', 'SprayBottle', 'Statue', 'Stool', 'TVStand', 'TeddyBear', 'Television', 'TennisRacket', 'TissueBox', 'Toaster', 'Toilet', 'ToiletPaper', 'ToiletPaperHanger', 'Tomato', 'Towel', 'TowelHolder', 'VacuumCleaner', 'Vase', 'WashingMachine', 'Watch', 'WateringCan', 'WineBottle'
    ),
    'palette': [
        (255, 216, 90), (226, 58, 157), (28, 86, 84), (186, 102, 20), (114, 117, 89), (199, 245, 189), (5, 201, 140), (171, 94, 0), (37, 56, 1), (197, 91, 186), (104, 145, 7), (164, 130, 184), (174, 158, 62), (184, 243, 178), (86, 248, 140), (214, 5, 74), (15, 185, 138), (134, 81, 233), (33, 208, 171), (230, 77, 247), (180, 138, 43), (36, 25, 79), (142, 6, 149), (25, 140, 152), (116, 248, 56), (86, 115, 220), (151, 241, 194), (57, 240, 121), (76, 38, 89), (46, 66, 102), (179, 176, 92), (175, 66, 200), (245, 203, 231), (164, 53, 35), (114, 105, 65), (247, 247, 171), (68, 225, 84), (241, 94, 191), (167, 194, 152), (16, 3, 75), (9, 85, 34), (70, 229, 14), (178, 248, 235), (184, 221, 128), (145, 248, 201), (31, 207, 98), (97, 175, 222), (83, 226, 88), (18, 145, 128), (244, 10, 50), (30, 218, 162), (111, 126, 187), (18, 176, 2), (151, 105, 6), (109, 180, 159), (97, 61, 44), (184, 155, 207), (125, 103, 57), (205, 8, 187), (107, 171, 191), (105, 216, 174), (50, 115, 252), (32, 219, 65), (84, 124, 66), (219, 215, 231), (107, 51, 216), (135, 110, 138), (231, 68, 123), (58, 35, 195), (25, 150, 90), (172, 216, 195), (129, 35, 241), (184, 127, 130), (143, 176, 89), (122, 98, 112), (101, 179, 205), (59, 105, 76), (143, 14, 61), (243, 75, 71), (170, 76, 64), (137, 126, 117), (64, 121, 168), (47, 188, 12), (106, 197, 76), (163, 91, 144), (37, 47, 69), (186, 188, 249), (23, 188, 154), (180, 83, 116), (2, 162, 43), (224, 116, 161), (86, 238, 232), (20, 229, 188), (29, 123, 144), (160, 247, 167), (8, 54, 243), (178, 94, 123), (253, 193, 1), (177, 130, 172), (169, 237, 91), (4, 162, 109), (205, 1, 125)
    ]
}
batch_size=4
num_workers=16

default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
# dataset_type = 'CocoDataset'
data_root = '..'
backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=True),
    dict(
        type='RandomResize', scale=[(1333, 640), (1333, 800)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='LoadProposalsFromAnnotations'),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=True),
    dict(type='LoadProposalsFromAnnotations'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type='CocoDataset',
            data_root='..',
            metainfo=metainfo,
            ann_file='train_dataset_info_per_asset.json',
            data_prefix=dict(img=''),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=[
                dict(type='LoadImageFromFile', backend_args=None),
                dict(
                    type='LoadAnnotations',
                    with_bbox=True,
                    with_mask=True,
                    poly2mask=True),
                dict(
                    type='RandomResize',
                    scale=[(1333, 640), (1333, 800)],
                    keep_ratio=True),
                dict(type='RandomFlip', prob=0.5),
                dict(type='LoadProposalsFromAnnotations'),
                dict(type='PackDetInputs')
            ],
            backend_args=None)))
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='..',
        metainfo=metainfo,
        ann_file='val_dataset_info_per_asset.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True,
                poly2mask=True),
            dict(type='LoadProposalsFromAnnotations'),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        backend_args=None))
test_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        data_root='..',
        ann_file='test_dataset_info_per_asset.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True,
                poly2mask=True),
            dict(type='LoadProposalsFromAnnotations'),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        backend_args=None))
val_evaluator = dict(
    type='CocoMetric',
    ann_file='../val_dataset_info_per_asset.json',
    metric=['bbox', 'segm'],
    backend_args=None)
test_evaluator = dict(
    type='CocoMetric',
    ann_file='../test_dataset_info_per_asset.json',
    metric=['bbox', 'segm'],
    backend_args=None,
    classwise=True)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[9, 11],
        gamma=0.1)
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))
auto_scale_lr = dict(enable=False, base_batch_size=batch_size*4)
model = dict(
    type='MaskRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=None,
    # rpn_head=dict(
    #     type='RPNHead',
    #     in_channels=256,
    #     feat_channels=256,
    #     anchor_generator=dict(
    #         type='AnchorGenerator',
    #         scales=[8],
    #         ratios=[0.5, 1.0, 2.0],
    #         strides=[4, 8, 16, 32, 64]),
    #     bbox_coder=dict(
    #         type='DeltaXYWHBBoxCoder',
    #         target_means=[0.0, 0.0, 0.0, 0.0],
    #         target_stds=[1.0, 1.0, 1.0, 1.0]),
    #     loss_cls=dict(
    #         type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
    #     loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=102,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=102,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    train_cfg=dict(
        rpn=None,
        rpn_proposal=None,        
        # rpn=dict(
        #     assigner=dict(
        #         type='MaxIoUAssigner',
        #         pos_iou_thr=0.7,
        #         neg_iou_thr=0.3,
        #         min_pos_iou=0.3,
        #         match_low_quality=True,
        #         ignore_iof_thr=-1),
        #     sampler=dict(
        #         type='RandomSampler',
        #         num=256,
        #         pos_fraction=0.5,
        #         neg_pos_ub=-1,
        #         add_gt_as_proposals=False),
        #     allowed_border=-1,
        #     pos_weight=-1,
        #     debug=False),
        # rpn_proposal=dict(
        #     nms_pre=2000,
        #     max_per_img=1000,
        #     nms=dict(type='nms', iou_threshold=0.7),
        #     min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=None,
        # rpn=dict(
        #     nms_pre=1000,
        #     max_per_img=1000,
        #     nms=dict(type='nms', iou_threshold=0.7),
        #     min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0),
            max_per_img=100,
            mask_thr_binary=0.5)))
