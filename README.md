# YOLOv2 using Caffe
This is an implementation for converting YOLOv2 from [DarkNet](http://pjreddie.com/darknet/) to Caffe, with the option to convert Leaky ReLU to default ReLU instead. It is a minimal fork of [this repo](https://github.com/nodefluxio/caffe-yolov2), and existing code has not been complemented with any docstrings or comments if they were missing.

For the original YOLOv2 paper, see: ["Redmon, Joseph, and Ali Farhadi. “YOLO9000: Better, Faster, Stronger." arXiv preprint arXiv:1612.08242 (2016)](https://arxiv.org/abs/1612.08242).

## What's inside
- YOLO to Caffe model converter (thanks to [Duangenquan](https://github.com/duangenquan/YoloV2NCS)).

## Supported models
Tiny YOLOv2: [cfg](https://raw.githubusercontent.com/pjreddie/darknet/e84933bfdd7315736c442a41d9aed163843dda54/cfg/yolov2-tiny.cfg) and [weights](https://pjreddie.com/media/files/tiny-yolo-voc.weights)

YOLOv2 without Reorg and Route layers, without weight conversion: [cfg](https://raw.githubusercontent.com/pjreddie/darknet/e84933bfdd7315736c442a41d9aed163843dda54/cfg/yolov2-voc.cfg)

This converter does not support `Route` and `Reorg` layers.

## Getting started
**NOTE** The following instructions assume that you already have a running Caffe v1.0 distribution in Python (see [here](http://caffe.berkeleyvision.org/installation.html) for instructions). Optionally you may make use of the `Dockerfile` included in this repo, see instructions for usage below.

1. Download the pre-trained YOLO models (config file `.cfg` and pre-trained weights `.weights`).

2. Convert the config file using `create_yolo_prototxt.py`:
```
python create_yolo_prototxt.py -c CFG_INPUT -m PROTOTXT_OUTPUT [--noleaky for replacing Leaky ReLU with ReLU activation]
```

3. Convert the pre-trained weights using `create_yolo_caffemodel.py`:
```
python create_yolo_caffemodel.py [-h] -m PROTOTXT_INPUT -w WEIGHTS_INPUT -o CAFFEMODEL_OUTPUT
``` 

### Setup with Docker
Build a Docker image with a minimal Caffe 1.0 installation, using the `Dockerfile` inside this repo:
```
docker build --network=host -t caffe-cpu18.04 .
```

Then create a Docker container based on this image and open a terminal inside:
```
docker run -v $PWD:/workspace --net=host -it caffe-cpu18.04 /bin/bash
cd /workspace
python3 create_yolo_prototxt.py [with arguments from above]
python3 create_yolo_caffemodel.py [with arguments from above]

```

## Disclaimer
Please note that unlike DarkNet, Caffe does not support default padding in their pooling layers. Depends on your model, you might find a difference in the output size. In our case of tiny-yolo-voc, Darknet produces 13x13x125 output whilst Caffe produces 12x12x125.

---
## Credits
This application uses Open Source components. You can find the source code of their open source projects below. We acknowledge and are grateful to these developers for their contributions to open source.

Project: [YoloV2NCS](https://github.com/duangenquan/YoloV2NCS) by duangenquan
for the YOLOv2 output parser and region parameter implementation

Project: [caffe-yolov2](https://github.com/nodefluxio/caffe-yolov2) by nodefluxio, serving as the baseline for the YOLOv2 parser.

Project: [darknet](https://github.com/pjreddie/darknet) by pjreddie, the framework YOLOv2 is originally implemented in.

## License
The code is released under the YOLO license.
