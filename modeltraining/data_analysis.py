#
# data_analysis.py
#
# Usage: data_analysis.py iternum solution -d (display graph, optional)
# Ex) python data_analysis.py 1 property
#
# Expects ./data_analysis 
#
# (!!!This folder is gitignored and must be manually added!!!)
#
# Given a generated dataset iteration number as well as
# property vs violent solution column, graph all present
# attributes relevant to the solution. Saves all generated
# graphs in a gitignored folder - this must be generated!
#
# Points are color-coded based on automatic deliniations
# to seperate data into equal portions. 

import argparse
import pandas as pd

import matplotlib
matplotlib.use("Agg") # To avoid Fail to allocate bitmap error when printing lots of graphs.
# See: https://github.com/matplotlib/mplfinance/issues/386
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap 
from autoviz.AutoViz_Class import AutoViz_Class
 
from variable_labels import VariableLabels

class DataAnalysis:
  variable_labels = None
  data_folder = "./complete"
  graph_folder = "./data_analysis"
  graph_folder_missing = False

  ignore_attributes = ["NAME","state_x","state_y","city","year_y","year_x","stateid","placeid"]

  def __init__(self):
    self.variable_labels = VariableLabels()

  # Given the dataset, execute autoviz on it, which happily and
  # automatically provides visualizations of the most important
  # features.
  def autoviz_dataset(self, iternum, solution):
    print("[INFO] Initializing AutoViz Data Analysis for iternum " + str(iternum) + " - " + str(solution) + ".")
    
    # Sanity check
    if(solution != "property" and solution != "violent"):
      print("[ERROR] Provided solution is not 'property' or 'violent'. Stopping...")
      return

    # Load data from csv file
    dataset_filename = "complete_iterA" + str(iternum) + "_" + solution + ".csv"
    dataset_path = self.data_folder + "/" + dataset_filename
    data = pd.read_csv(dataset_path)
    if data is None:
      print("[ERROR] Unable to read " + dataset_path + "! Stopping...")
      return
    print("[INFO] Data read from " + dataset_path + " successfully.")

    # In order for the variable names to be replaced properly in
    # the output of the autoviz, let's rename them all now.
    """
    column_name_mapper = {} 
    for column in data:
      new_name = self.variable_labels.fetch_label(column)
      column_name_mapper[column] = new_name
    data = data.rename(columns=column_name_mapper)
    """
    
    # All done, let's execute autoviz. 
    av = AutoViz_Class()
    dataset_name = "A" + str(iternum) + " " + solution
    dft = av.AutoViz(depVar=solution, verbose=2, dfte=data, lowess=False, filename="", chart_format="svg", save_plot_dir=self.graph_folder + "/" + dataset_name)

  # Given the dataset, graph a scatter plot for every attribute
  def scatter_dataset(self, iternum, solution, display_graphs = False):
    print("[INFO] Initializing Scatterplot Data Analysis for iternum " + str(iternum) + " - " + str(solution) + ".")

    # Sanity check
    if(solution != "property" and solution != "violent"):
      print("[ERROR] Provided solution is not 'property' or 'violent'. Stopping...")
      return
    
    # Load data from csv file
    dataset_filename = "complete_iterA" + str(iternum) + "_" + solution + ".csv"
    dataset_path = self.data_folder + "/" + dataset_filename
    data = pd.read_csv(dataset_path)
    if data is None:
      print("[ERROR] Unable to read " + dataset_path + "! Stopping...")
      return
    print("[INFO] Data read from " + dataset_path + " successfully.")

    # We need to seperate data into sections for our plotter. 
    # We'll seperate the crime rates into quartiles.
    # For visualization, we'll use the ratio of crime over 
    # population.
    solution_list = []
    for i in range(0, len(data[solution])):
      solution_list.append(float(data[solution][i])/float(data["population"][i]))
    solution_col = pd.DataFrame(solution_list, columns=["solution"])["solution"]
    solution_min = solution_col.min()
    solution_max = solution_col.max()
    solution_q1 = solution_col.quantile(0.25)
    solution_q2 = solution_col.quantile(0.5)
    solution_q3 = solution_col.quantile(0.75)

    boundaries = [solution_min, solution_q1, solution_q2, solution_q3, solution_max]
    colors = ["green","blue","orange","red"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, len(colors))
    c = solution_col

    # Output data statistics and plot!
    data_rows, data_cols = data.shape 
    print("[INFO] " + str(data_rows) + " instances, " + str(data_cols)+ " columns.")
    print("[INFO] Boundaries of data are: ", end="")
    print(boundaries, end=".\n")
    dataset_name = "A" + str(iternum) + " " + solution
    for col in data:
      if self.graph_folder_missing is True:
        print("[ERROR] Unable to save generated graph! Ensure " + self.graph_folder + " folder is present!")
        return
      # Don't graph things that aren't useful, like NAME.
      if col not in self.ignore_attributes:
        self.scatter_attribute(data=data, dataset_name=dataset_name, attribute=col, c=c ,cmap=cmap, norm=norm, display_graphs=display_graphs)

  # Given a single attribute + other data, graph it. 
  def scatter_attribute(self, data, dataset_name, attribute, c=None, cmap=None, norm=None, display_graphs = False):
    # The attribute code (Ex) DP02_0001E) tells us little. Use
    # the map we read from the initial file to expand it. 
    variable_label = self.variable_labels.fetch_label(attribute)
    title = dataset_name + " - " + variable_label
    print("[INFO] Graphing: " + attribute + " - " + variable_label)

    title_fontsize = 8 # Some variable names are a mouthful. 
    graph_width_inches = 13
    graph_height_inches = 7

    # Generate the graph.
    fig = plt.figure(1)
    fig.suptitle(title, fontsize = title_fontsize)
    fig.set_size_inches(graph_width_inches,graph_height_inches)
    plt.scatter(data.index, data[attribute], c=c, cmap=cmap, norm=norm)
    if display_graphs:
      plt.show()

    # Save the graph.
    try:
      fig.savefig(self.graph_folder + "/" + dataset_name + "/scatter_" + attribute)
    except:
      self.graph_folder_missing = True

    plt.close("all")

if __name__ == "__main__":

  iternum = None
  solution = None
  display_graphs = None
  use_autoviz = None

  try: 
    parser = argparse.ArgumentParser()
    parser.add_argument("iternum")
    parser.add_argument("solution")
    parser.add_argument("-d", required=False, action="store_true", default=False)
    parser.add_argument("-a", required=False, action="store_true", default=False)
    args = parser.parse_args()

    iternum = args.iternum
    solution = args.solution
    display_graphs = args.d
    use_autoviz = args.a
  except:
    # We're running in debug.
    iternum = 1
    solution = "property"
    display_graphs = True
    use_autoviz = False

  data_analysis = DataAnalysis()
  if use_autoviz:
    data_analysis.autoviz_dataset(iternum,solution)
  else:
    data_analysis.scatter_dataset(iternum, solution, display_graphs)
  