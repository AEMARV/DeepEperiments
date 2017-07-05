from keras.utils.vis_utils import plot_model
from utils import global_constant_var
# from keras.utils.visualize_util import plot
import pickle
import os
class HistoryHolder():
	#CONSTANTS
	#END_OF_CONSTANTS
	#FIELDS:
	# dir_abs_path='' # After init is /FOLDER_NAME_RESULTS/
	# plots_abs_path=''
	# figure_abs_path = ''
	# metric_plot_container=None
	# weight_visualizer_container = None
	# experiment_name = "TEST"
	# experiment_index = '0'
	#END_OF_FIELDS
	def __init__(self,experiment_name,opts,relative_result_path=global_constant_var.PROJECT_RELATIVE_PATH):
		self.experiment_name = experiment_name
		self.relative_result_path = os.path.join(relative_result_path, global_constant_var.FOLDER_NAME_RESULTS,opts['experiment_tag'],
		experiment_name)
		self.folder_creation_wrapper()
		# self.metric_plot_container=PlotContainer(self.plots_abs_path, self.experiment_index)
		self.opts=opts
	def folder_creation_wrapper(self):
		self.dir_abs_path = os.path.abspath(self.relative_result_path)
		self.create_main_folder_hierachy(self.relative_result_path)
		self.create_expriment_index()
		self.plots_abs_path = os.path.join(self.dir_abs_path,global_constant_var.FOLDER_NAME_PLOTS)
		global_constant_var.Figure_abs_path = os.path.join(self.dir_abs_path, global_constant_var.FOLDER_NAME_FIGURES)
	def create_expriment_index(self):
		i=1
		abs_path = self.dir_abs_path
		while True:
			if os.path.exists(os.path.join(abs_path,str(i))):
				i=i+1
			else:
				os.mkdir(os.path.join(abs_path,str(i)))
				self.experiment_index=str(i)
				break
		self.dir_abs_path = os.path.join(abs_path,self.experiment_index)


	def create_main_folder_hierachy(self,relative_result_path):
		abs_path = os.path.abspath(relative_result_path)
		self.createFolderPath(abs_path)

	def createFolderPath(self,abs_path):
		# making sure all the folders in the path exist o.w. creating those folders
		folder_list = abs_path.split('/')
		path_to_be_created = ''
		for folder in folder_list[1:]:
			path_to_be_created += '/' + folder
			if not os.path.exists(path_to_be_created):
				os.mkdir(path_to_be_created)
	def model_plot(self,model):
		plot_model(model, show_shapes=True,to_file=self.dir_abs_path+'/model.png')
		with open(self.dir_abs_path+'/model_config.txt', 'w') as f:
			f.write(str(model.get_config()))
		with open(self.dir_abs_path+'/model_config.yaml', 'w') as f:
			f.write(str(model.to_yaml()))
		with open(self.dir_abs_path + '/model_config.json', 'w') as f:
			f.write(str(model.to_json()))
	def store_opts(self):
		with open(self.dir_abs_path+'/opts.pkl', 'wb') as f:
			pickle.dump(self.opts, f, 0)
		with open(self.dir_abs_path+'/opts.txt', 'w') as g:
			g.write(str(self.opts))

def test_HH(opts):
	hh = HistoryHolder(experiment_name="test_hh", opts=opts)
	# hh.metric_plot_container.create_random_plots()
if __name__ == '__main__':
    hh = HistoryHolder(experiment_name="test_hh",opts=None)
    # hh.metric_plot_container.create_random_plots()
    print hh.relative_result_path
