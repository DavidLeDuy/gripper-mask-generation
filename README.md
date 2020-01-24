# Introduction

Based on [Mask RCNN](https://github.com/matterport/Mask_RCNN), we developed a gripper-mask-generation programm.
You could clone it and try it out. Of course you can also train on your own data.
Welcome to ask questions on the issues, we would like to share our experience to you.
Thanks again for the great work of [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN) !

# Installation

1. Clone this repository
2. Run setup.sh
   ```
   sh setup.sh
   ```
3. Run setup.py from the repository root directory
   ```
   python3 setup.py install
   ```
4. Download pre-trained COCO weights (mask_rcnn_coco.h5) from the [release page](https://github.com/matterport/Mask_RCNN/releases).
5. Download pre-trained Gripper weights (XXX) from the [release page](XXXXX).

# Run Gripper-mask

1. Go to gripperAI directory
   ```
   cd gripperAI
   ```
2. Detect test_img1.png
   ```
   python gripperAI_final.py splash --weights=gripperAImodel_final.h5 --image=test_img1.png
   ```
3. Detect test_video.mp4
   `python gripperAI_final.py splash --weights=gripperAImodel_final.h5 --video=test_video.mp4`
   ![bbox_demo](</gripperAI/demo(img+video)/20191009T155539_boundingbox.png>)
   ![mrcnn_demo](</gripperAI/demo(img+video)/splash_20191009T093708.png>)

# Train your own dataset

1. Run settings.py
   ```
   python3 settings.py
   ```
2. Use [VGG Annotation](http://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html) to label train/val data. settings.py creates a folders for your data. Suggested ratio of "train"/"val" = 8/2. And export annotations as JSON. Currently works only with polygon.
3. Rename JSON file as 'via_region_data.json' for both '.json' in the "train" and "val" folders.
4. Reduplicate gripperAI_final.py and rename it to your usage.
5. Go through your mrcnn python file and modify the code as indicated in the file.
6. Finally you can start to train on your own dataset. Our default pre-trained model is 'coco.'
   ```
   python XXXX.py train --dataset=/PATH/TO/dataset --weights=coco
   ```
7. After training, you can run your python file again with "splash" command
