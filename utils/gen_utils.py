import os
from utils import global_constant_var
def find_key_value_to_str_recursive(dictionary,parent_dict_name,method_name_exclusive_list):
	result_str = ''
	for i in dictionary.items():
		if (i[0]=='method' or i[0]=='rate') and method_name_exclusive_list.__contains__(parent_dict_name):
			result_str+='#'+(parent_dict_name+':'+str(i[1]))
		if type(i[1])==dict:
			result_str+=find_key_value_to_str_recursive(i[1],i[0],method_name_exclusive_list)
	return result_str
def get_experiment_name_prompt(substitute_prompt=None):
	prompting = "please select the Experiment name or Define a new one" if substitute_prompt else substitute_prompt
	print(substitute_prompt)
	dirs = [d for d in os.listdir('./Results') if os.path.isdir(os.path.join('./Results', d)) and not d[0] == '.']
	dirs = [d for d in dirs]
	a = ['[' + str(index) + ']' + d + ' \n' for index, d in enumerate(dirs)]
	index = len(dirs)
	a = a + ['[' + str(index + 1) + ']' + 'New Experiment']
	print((''.join(a)))
	experiment_index = input('Please type in the index of option \n')
	if experiment_index == '':
		experiment_name = 'Debug'
	else:
		if int(experiment_index) == index + 1:
			experiment_name = input('Type in Experiment Name \n')
			experiment_description = input('Please Type in Experiment Description \n')
		else:
			experiment_name = dirs[int(experiment_index)]
			experiment_description = ''
	return experiment_name
def get_model_from_experiment_prompt(experiment_name,dataset_str):
	print("please select the Model name ")
	root = os.path.join('.','Results',experiment_name,dataset_str)
	dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and not d[0] == '.']
	dirs = [d for d in dirs]
	a = ['[' + str(index) + ']' + d + ' \n' for index, d in enumerate(dirs)]
	index = len(dirs)
	a = a + ['[' + str(index + 1) + ']' + 'New Experiment']
	print((''.join(a)))
	model_index = input('Please type in the index of option \n')
	model_name = dirs[int(model_index)]
	return model_name