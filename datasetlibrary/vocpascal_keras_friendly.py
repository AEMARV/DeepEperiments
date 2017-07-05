from __future__ import absolute_import
import os
import shutil

def createFolderPath(abs_path):
	# making sure all the folders in the path exist ow creating those folders
	folder_list = abs_path.split('/')
	path_to_be_created = ''
	for folder in folder_list[1:]:
		path_to_be_created += '/' + folder
		if not os.path.exists(path_to_be_created):
			os.mkdir(path_to_be_created)

def copyDatasetFlowDirectory(anntn_abs_path,anntn_file_detect_str,src_abs_path,dst_abs_path):
	# copies files from VOC into the specified folder with regards to annotations(creates one folder for each class)
	#  anntn_file_detect_str makes the function read the annotation files with the unique string signature egg
	# "_train." will read only train files
	anntn_dict = os.listdir(anntn_abs_path)
	for file_anntn in anntn_dict:
		if file_anntn.find(anntn_file_detect_str) == -1:
			continue
		classname = file_anntn.split('_')[0]
		class_abs_path_keras_dest = os.path.join(dst_abs_path, classname)
		if not os.path.exists(class_abs_path_keras_dest):
			os.mkdir(class_abs_path_keras_dest)
		file_anntn_path = os.path.join(anntn_abs_path, file_anntn)
		anntn_file = open(file_anntn_path)
		for line in anntn_file:
			attr = line.split()
			if (attr[1] == '1'):
				image_file_name = attr[0] + ".jpg"
				target_image_path_src = os.path.join(src_abs_path, image_file_name)
				target_image_path_dst = os.path.join(class_abs_path_keras_dest, image_file_name)
				if not os.path.exists(target_image_path_dst):
					shutil.copy2(src=target_image_path_src, dst=class_abs_path_keras_dest)

if __name__ == '__main__':
	# Path Initialization
	IMAGE_REL_PATH = "../../Datasets/VOC/VOCdevkit/VOC2012/JPEGImages"
	ANNOTATION_REL_PATH = "../../Datasets/VOC/VOCdevkit/VOC2012/ImageSets/Main"
	TRAINING_REL_PATH_KERAS_DEST = "../../Datasets/VOC/Keras/training"
	VALIDATION_REL_PATH_KERAS_DEST = "../../Datasets/VOC/Keras/validation"
	TRAINVAL_REL_PATH_KERAS_DEST = "../../Datasets/VOC/Keras/trainvalidation"
	image_abs_path = os.path.abspath(IMAGE_REL_PATH)
	anntn_abs_path = os.path.abspath(ANNOTATION_REL_PATH)
	training_abs_path_keras_dest = os.path.abspath(TRAINING_REL_PATH_KERAS_DEST)
	validation_abs_path_keras_dest = os.path.abspath(VALIDATION_REL_PATH_KERAS_DEST)
	trainval_abs_path_keras_dest = os.path.abspath(TRAINVAL_REL_PATH_KERAS_DEST)

	# copying dataset
	createFolderPath(training_abs_path_keras_dest)
	createFolderPath(validation_abs_path_keras_dest)
	createFolderPath(trainval_abs_path_keras_dest)
	copyDatasetFlowDirectory(anntn_abs_path, "_train.", image_abs_path, training_abs_path_keras_dest)
	copyDatasetFlowDirectory(anntn_abs_path, "_val.", image_abs_path, validation_abs_path_keras_dest)
	copyDatasetFlowDirectory(anntn_abs_path, "_trainval.", image_abs_path, trainval_abs_path_keras_dest)
