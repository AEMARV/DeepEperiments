import numpy as np
from matplotlib import pyplot as plt
import os

class PlotContainer():
	FIG_Handle = "fig_handle"
	LINE_HANDLES_DICT = "line_handle_dict"
	# e.g. fig_handles = { fig_name:{    FIG_HANDLE:fig_handle   ,   LINE_HANDLES_DICT:  {  line_labeli:lineHandle_i
	# }  }
	# fig_handles[some_fig_name][FIG_HANDLE|LINE_HANDLE_DICT]

	#FIELDS:
	result_dir_abs_path = ''
	figure_handles = {}
	container_id_string = ''
	#END_OF_FIELDS


	def __init__(self,result_dir_abs_path,container_id_string):
		# type: (object, object) -> object
		self.result_dir_abs_path = result_dir_abs_path
		self.container_id_string=container_id_string
		assert os.path.exists(self.result_dir_abs_path)
		plt.ion()

	def figure_add(self, name_fig):
		assert self.figure_handles.get(name_fig) == None
		self.figure_handles[name_fig] = {
			self.FIG_Handle: plt.figure(name_fig),
			self.LINE_HANDLES_DICT: {}
			}

	def figure_select(self, name_fig):
		"""return figure with name from figure_handles dict and selects the figure as current figure"""
		assert self.figure_handles.get(name_fig) != None
		plt.figure(name_fig)
		return self.figure_handles[name_fig][self.FIG_Handle]

	def line_add(self, name_fig, name_label):
		""" selects the figure and add a legend to it"""
		assert self.figure_handles[name_fig][self.LINE_HANDLES_DICT].get(name_label) == None
		self.figure_select(name_fig)
		line_handle, = plt.plot([], [], label=name_label)
		plt.legend()
		self.figure_handles[name_fig][self.LINE_HANDLES_DICT][name_label] = line_handle

	def line_append_point(self, name_fig, name_label, x, y):
		"""append a point to data with name_label contained in name_fig figure"""
		self.figure_select(name_fig)
		line = self.figure_handles[name_fig][self.LINE_HANDLES_DICT][name_label]
		line.set_xdata(np.append(line.get_xdata(), x).tolist())
		line.set_ydata(np.append(line.get_ydata(), y).tolist())
		plt.plot(line.get_xdata(), line.get_ydata(), 'r*')

	def add_line_to_all_figs(self, name_label_list):
		"""add a legend to all figures in contained in this object"""
		for fig_name in self.figure_handles.keys():
			for label_name in name_label_list:
				self.line_add(fig_name, name_label=label_name)

	def add_figure_from_list(self, fig_name_list):
		for fig_name in fig_name_list:
			self.figure_add(fig_name)

	def add_fig_lines_from_list(self, name_fig_list, name_label_list):
		self.add_figure_from_list(name_fig_list)
		self.add_line_to_all_figs(name_label_list)
	def save_fig_as_png(self,name_fig,path_abs=None):
		if path_abs == None:
			path_abs = self.result_dir_abs_path

		self.figure_select(name_fig)
		abs_path=os.path.join(path_abs,name_fig)
		plt.savefig(abs_path)
	def save_all_fig(self,path_abs=None):
		if path_abs==None:
			path_abss=self.result_dir_abs_path
		for name_fig in self.figure_handles.keys():
			self.save_fig_as_png(name_fig,path_abss)
	def create_random_plots(self):
		self.add_fig_lines_from_list(["figa", "figb", "figc"], ["train", "val"])
		self.line_append_point("figa", "train", [1, 5, 7], [2, 3, 6])
		self.line_append_point("figb", "train", [3, 6, 3], [4, 2, 1])
		self.line_append_point("figc", "val", [3, 5, 6], [5, 1, 7])
		self.line_append_point("figa", "train", [2, 3, 4], [2, 3, 6])
		self.save_all_fig()
if __name__ == '__main__':
	abs_path = os.path.abspath('../Results/')
	pc = PlotContainer(abs_path, 0)
	pc.create_random_plots()
	plt.pause(10)
