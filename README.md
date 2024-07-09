# <div align="center">easy-YOLOv5</div>

### Description

This is a repository for implementation of YOLOv5 for easy customization and understanding underlying techniques in it, which is refered to ultralytics' YOLOv5 (https://github.com/ultralytics/yolov5).   


### User Command 

You can train your own YOLOv5 model with command like below. As for <DATASET> you can refer sample file in cfg/*.yaml, and make <DATASET NAME>.yaml file following your dataset. Since cfg/*.json file that is required to compute mAP scores is built automatically via dataloader, you do not have to worry about it. 


 - **Pretrained Model Weights Download**

	- [YOLOv5-n/s/m/l/x](https://drive.google.com/drive/folders/1cMiAjhkb9tWFxGtxf6WwqgBAxs58HOfX?usp=sharing)

| Model | Dataset | Train | Valid | Size<br><sup>(pixel) | mAP<br><sup>(@0.5:0.95) | mAP<br><sup>(@0.5) | Params<br><sup>(M) | FLOPs<br><sup>(B) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| YOLOv5n | COCO | train2017 | val2017 | 640 | 28.0 | 45.7 | 1.9 | 4.5 |
| YOLOv5s | COCO | train2017 | val2017 | 640 | 37.4 | 56.8 | 7.2 | 16.5 |
| YOLOv5m | COCO | train2017 | val2017 | 640 | 45.4 | 64.1 | 21.2 | 49.0 |
| YOLOv5l | COCO | train2017 | val2017 | 640 | 49.0 | 67.3 | 46.5 | 109.1 |
| YOLOv5x | COCO | train2017 | val2017 | 640 | 50.7 | 68.9 | 86.7 | 205.7 |


```python

# Training
python train.py --arch yolov5n --img-size 640 --num-epochs 200 --mosaic --cos-lr --model-ema --project <YOUR PROJECT> --dataset <YOUR DATASET>

# Evaluation
python val.py --project <YOUR PROJECT>

# Inference in images
python test.py --project <YOUR PROJECT> --test-dir <IMAGE DIRECTORY>

# Inference in video
python infer.py --project <YOUR PROJECT> --vid_path <VIDEO PATH>
```

---
## [Contact]
- Author: Jiho Park  
- Email: pjh5672.dev@gmail.com  