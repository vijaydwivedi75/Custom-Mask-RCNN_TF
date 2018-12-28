# Detecting Custom Objects with Mask RCNN using TensorFlow

Mask RCNN is used in Object Detection to predict instances (masks) of objects present in an image. Using TensorFlow Object Detection API and its pre-trained models, we can easily have our own object detection tool ready and setup in less time than you could ever expect.

# Steps

## Set up directories and TF Models
Create a root directory with a suitable name (say, ObjectDetection_MaskRCNN).

 1. cd to the root directory: 
 `cd ./ObjectDetection_MaskRCNN`
 2. Download TensorFlow code base and models
	 `git clone https://github.com/tensorflow/models.git`
3. After finishing download, cd to the models/research directory
   `cd ./models/research/` 
4. Run these commands to set up environment and add your `pwd` to Python Path. *Note: you have to run these commands whenver you start a terminal and want to use TF models.*
``export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim``
``protoc object_detection/protos/*.proto --python_out=.``

## Creating custom data and preparation

For our custom task, we try to detect Red Stamps in attested documents. We download ~70 publicly accesible attested documents from Google Images (*multiple sources*) and use it for the training and validation purposes.


 - Here is a snapshot of the data.
 ![sample of attested documents dataset](https://i.imgur.com/peHRpQt.jpg)
 - Divide the images to `JPEGImages` for training and `testImages` for testing
 - The `dataset/` folder should be in the root directory `ObjectDetection_MaskRCNN/` and should have the following directory setup.
	 - Annotations
		 - masks
		 - xmls
		 - *< color masks files >*
	 - JPEGImages
		 - *< training image files >*
	 - testImages
		 - *< test image files >*
	 - label.pbtxt
	 - train.record
 - Use [PixelAnnotationTool](https://github.com/abreheret/PixelAnnotationTool/releases) to create object masks. The instruction to mask an image can be seen [here](https://www.youtube.com/watch?v=wxi2dInWDnI).
 - You will get 3 outputs for every image while using the PixelAnnotationTool
	 - IMAGE_mask.png
	 - IMAGE_color_mask.png
	 - IMAGE_watershed_mask.png
		![enter image description here](https://i.imgur.com/z87O8XV.png =500x)
 - Rename all IMAGE_watershed_mask.png to IMAGE_mask.png and place inside `.dataset/Annotations/masks` folder.
 - Rename all IMAGE_color_mask.png to IMAGE.png and place inside `./dataset/Annotations` folder
 - Convert your data to TF Record format (*to generate the `train.record` file*)
	 -  Place the `'create_mask_rcnn_tf_record.py` from the directory `Imp_Files` to `models/research/object_detection/dataset_tools/` directory.
	 - To edit class name as per you requirement, edit this python script at line 57.
	 - From the directory `./models/research/` as present working directory, run the following command
	 `py object_detection/dataset_tools/create_mask_rcnn_tf_record.py \`
	 `--data_dir=/home/vijay/ObjectDetection_MaskRCNN/dataset \`
	 `--annotations_dir=Annotations \`
	 `--image_dir=JPEGImages \`
	 `--output_dir=/home/vijay/ObjectDetection_MaskRCNN/dataset/train.record \`
	 `--label_map_path=/home/vijay/ObjectDetection_MaskRCNN/dataset/label.pbtxt`

- You will see a `train.record-xxx` file inside the `dataset` directory. Rename it to `train.record`

## Training

 - Create 3 folders inside the root directory `IG`, `CP` and `pre_trained_models`. The current project structure would look like.
	 - ObjectDetection_MaskRCNN/
		 - CP/
		 - IG/
		 - dataset/
		 - pre_trained_models/
- Download any model starting with the name `mask_rcnn` from the [TensorFlow Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) and extract the compressed file inside the `pre_trained_models` folder. For this project, we download `mask_rcnn_inception_v2_coco`
- Copy the file `mask_rcnn_inception_v2_coco.config` file from `./models/research/object_detection/samples/configs` to the root directory `ObjectDetection_MaskRCNN/` and edit PATH_TO_BE_CONFIGURED at 5 locations inside this config file. Refer to the config file in this repo for making changes. You can further explore other configurations and adjust as per your needs.
- Finally, run this command from `./models/research` as present working directory to start training.
`py object_detection/legacy/train.py \`
`--train_dir=/home/vijay/ObjectDetection_MaskRCNN/CP \`
 `--pipeline_config_path=/home/vijay/ObjectDetection_MaskRCNN/mask_rcnn_inception_v2_coco.config`
 - Keep training for a few hundred (or thousand steps) and/or till the loss comes near to `0.1 or 0.2` or something.
## Export Inference Graph and Run for Test Images
- Check the step on which a `.ckpt` file has been saved. Let's say, the step is 230. Run the following command from `./models/research` as present working directory to save frozen inference graph.
`py object_detection/export_inference_graph.py \`
`--input_type=image_tensor \`
`--pipeline_config_path=/home/vijay/ObjectDetection_MaskRCNN/mask_rcnn_inception_v2_coco.config \`
`--trained_checkpoint_prefix=/home/vijay/ObjectDetection_MaskRCNN/CP/model.ckpt-230 \` 
`--output_directory=/home/vijay/ObjectDetection_MaskRCNN/IG`
- Run the notebook `mask_rcnn_eval.ipynb` to test on the images you have kept aside in the `testImages` folder. Explore the notebook and make the changes as you understand.
- Please note to make changes in the `visualization_utils.py` file inside `./models/research/object_detection/utils` folder to change the way you want to visualize the bounding boxes and object masks.
![enter image description here](https://i.imgur.com/ceN8b10.png)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE3ODgzMzkzNjldfQ==
-->