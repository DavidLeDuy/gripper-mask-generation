# Introduction

Based on [Mask RCNN](https://github.com/matterport/Mask_RCNN), we developed a easy to use mask-generation programm.
Just clone it and try it out. You can follow the steps to train on your own dataset.

# Installation

1. Clone this repository (duh)
2. Run setup.sh
   ```
   sh setup.sh
   ```
   This creates the folder structure and a virtual environment "venv" and installs the needed modules.
   ```
   gripper-mask-generator
   │   README.md
   │   setup.sh
   │   setup.py
   │   ...
   └───project
   │   │   mask_rcnn.py
   │   │
   │   └───models
   │   │    └─── [project name + timestamp]
   │   │    └─── ...
   │   │
   │   └─── data
   │        └─── test
   |        |     | [image name].png
   |        |     | ...
   |        |     | via_region_data.json
   │        └─── train
   |        |     | [image name].png
   |        |     | ...
   |        |     | via_region_data.json
   |        └─── val
   └─── ...
   ```

# Run Gripper-mask _(optional)_

1. Download pre-trained Gripper weights (XXX) from the [release page](XXXXX).
2. Go to project directory
   ```
   cd project
   ```
3. Detect test_img1.png
   ```
   python gripperAI_final.py splash --weights=gripperAImodel_final.h5 --image=test_img1.png
   ```
4. Detect test_video.mp4
   `python gripperAI_final.py splash --weights=gripperAImodel_final.h5 --video=test_video.mp4`
   ![bbox_demo](</project/gripperAI/demo(img+video)/20191009T155539_boundingbox.png>)
   ![mrcnn_demo](</project/gripperAI/demo(img+video)/splash_20191009T093708.png>)

# Train your own dataset

1. Download pre-trained COCO weights (mask_rcnn_coco.h5) from the [release page](https://github.com/matterport/Mask_RCNN/releases) and place it into models folder.
2. Run settings.py
   ```
   python3 settings.py
   ```
3. Use [VGG Annotation](http://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html) to label train/val data. Suggested ratio of "train"/"val" = 8/2. And export annotations as JSON.
    Region types: Rectangle, Polygon, Circle, Ellipse are supported.
4. Rename JSON file as 'via_region_data.json' for both '.json' in the "train" and "val" folders.
5. Finally you can start to train on your own dataset. The default pre-trained model is 'coco.'
   ```
   python [project_name].py train --dataset=/PATH/TO/dataset --weights=coco
   ```
6. After training, you can run your python file again with "splash" command
   ```
   python [project_name].py splash --weights=[your weight path] --image=[image path]
   ```
