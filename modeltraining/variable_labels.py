#
# variable_labels.py
#
# Given a particular ACS variable, return the label. If the
# variable is not found, or if the csv file was not found
# or read, the variable name is simply returned. 
#
# Expects: acs/variable_labels.xlsx
#
# Ex) In:  DP02_0001E
#     Out: Estimate!!HOUSEHOLDS BY TYPE!!Total households

import pandas as pd

class VariableLabels:
  variable_labels_file = "./acs/variable_labels.csv"
  variable_labels = None

  # On initialization, load the CSV and process into a dict.
  #
  # Note, for now we don't pay attention to the other info
  # the ACS provides, i.e. Conept, Required, Attributes,
  # Limit, Predicate, Group. 
  def __init__(self):
    data = pd.read_csv(self.variable_labels_file)
    variables = data["Name"]
    labels = data["Label"]

    self.variable_labels = {}
    for i in range(0, len(variables)):
      self.variable_labels[variables[i]] = labels[i]

  # On demand, given a variable, return either the label of
  # the variable or the variable itself if not found or there
  # was a csv read error. 
  def fetch_label(self, variable):
    if self.variable_labels is not None and variable in self.variable_labels:
      return self.variable_labels[variable]
    else:
      return variable