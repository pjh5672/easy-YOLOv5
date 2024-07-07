python train.py --arch yolov5m --img-size 448 --num-epochs 100 --mosaic --cos-lr --model-ema --project brain-m-448 --dataset brain

python train.py --arch yolov5n --img-size 448 --num-epochs 100 --mosaic --cos-lr --model-ema --project catdog-n-448 --dataset catdog

python train.py --arch yolov5n --img-size 448 --num-epochs 100 --mosaic --cos-lr --model-ema --scratch --project catdog-n-448-scratch --dataset catdog

python train.py --arch yolov5s --img-size 448 --num-epochs 100 --mosaic --cos-lr --model-ema --project drive-s-448 --dataset drive


