import os.path as path
import os
PYTHON_PROJECT_NAME ='DeepEperiments'
FOLDER_NAME_RESULTS="Results"
FOLDER_NAME_FIGURES="Figures"
FOLDER_NAME_PLOTS="Plots"
PROJECT_ABS_PATH = path.abspath('.')

while True:
	SPLIT_PATH = path.split(PROJECT_ABS_PATH)
	if SPLIT_PATH[1]==PYTHON_PROJECT_NAME:
		PROJECT_RELATIVE_PATH = path.relpath(PROJECT_ABS_PATH)
		break
	else:
		PROJECT_ABS_PATH = SPLIT_PATH[0]
	if PROJECT_ABS_PATH== '/':raise ValueError('Project folder cannot be found')
del SPLIT_PATH
def get_experimentcase_abs_path(experiment_tag,dataset,model_str,model_desc=None,run_num=1):
	if model_desc is not None:
		return path.join(PROJECT_ABS_PATH,experiment_tag,dataset,model_str,model_desc,run_num)
	else:
		model_dir = path.join(PROJECT_ABS_PATH,FOLDER_NAME_RESULTS, experiment_tag, dataset, model_str)
		dir_list = os.listdir(model_dir)
		for d in dir_list:
			if os.path.isdir(os.path.join(model_dir, d)): return path.join(model_dir,d,str(run_num))
