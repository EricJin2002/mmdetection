# MMDetection魔改版

## single-view object recognition

### test

```sh
python tools/test.py checkpoints/customized_mask-rcnn_r101_fpn_ms-poly-3x_coco.py checkpoints/mask_rcnn_r101_fpn_mstrain-poly_3x_coco_20210524_200244-5675c317.pth
```

### fine-tune

#### 1. 修改配置文件中`load_from`的值

#### 2. 运行

```sh
python tools/train.py checkpoints/customized_mask-rcnn_r101_fpn_ms-poly-3x_coco.py --auto-scale-lr
```

### 备忘

以下命令在后台执行 root 目录下的 runoob.sh 脚本，并重定向输入到 runoob.log 文件：

```sh
nohup /root/runoob.sh > runoob.log 2>&1 &
```