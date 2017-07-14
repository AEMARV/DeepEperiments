import numpy as np
from matplotlib import pyplot as plt
import os

class FigureContainer(object):
	FIG_Handle = "fig_handle"
	# e.g. fig_handles = { fig_name:{    FIG_HANDLE:fig_handle   ,   LINE_HANDLES_DICT:  {  line_labeli:lineHandle_i
	# }  }
	# fig_handles[some_fig_name][FIG_HANDLE|LINE_HANDLE_DICT]

	#FIELDS:
	# result_dir_abs_path = ''
	# figure_handles = {}
	# container_id_string = ''
	#END_OF_FIELDS


	def __init__(self,result_dir_abs_path,container_id_string):
		# type: (object, object) -> object
		self.result_dir_abs_path = result_dir_abs_path
		self.container_id_string=container_id_string
		self.figure_handles={}
		assert os.path.exists(self.result_dir_abs_path)
		plt.ion()

	def figure_add(self, name_fig):
		assert self.figure_handles.get(name_fig) == None
		self.figure_handles[name_fig] = {
			self.FIG_Handle: plt.figure(name_fig,frameon=False),
			}
	def exist_figure(self,name_fig):
		if self.figure_handles.get(name_fig) != None:
			return True
		else:
			return False

	def figure_select(self, name_fig):
		"""return figure with name from figure_handles dict and selects the figure as current figure"""
		assert self.figure_handles.get(name_fig) != None
		plt.figure(name_fig)
		return self.figure_handles[name_fig][self.FIG_Handle]

	def add_figure_from_list(self, fig_name_list):
		for fig_name in fig_name_list:
			self.figure_add(fig_name)

	def save_fig_as_png(self,name_fig,path_abs=None):
		if path_abs == None:
			path_abs = self.result_dir_abs_path

		self.figure_select(name_fig)
		abs_path=os.path.join(path_abs,name_fig)
		plt.savefig(abs_path)
	def save_all_fig(self,path_abs=None):
		if path_abs==None:
			path_abss=self.result_dir_abs_path
		for name_fig in list(self.figure_handles.keys()):
			self.save_fig_as_png(name_fig,path_abss)
