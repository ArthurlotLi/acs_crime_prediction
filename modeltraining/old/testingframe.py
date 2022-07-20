# Testing frame for calling both autodataset.py
# to generate the dataset as well as neuralnetwork_base.py
# and to return the mae solutions of all given options for
# both. 

import pandas as pd 

from autodataset import autodatasetgenerator
from neuralnetwork_base import neuralnetwork

seperationtext = "============================================================\n============================================================"
EPOCHS = 3000
testiternum = 'tf'
propertycsv = 'complete_iterA' + testiternum + '_property.csv'
violentcsv = 'complete_iterA' + testiternum + '_violent.csv'
acs = "acscomplete.csv"

# POPULATION THRESHOLD
def test_A():
  print("Initializing testA()! This might take awhile.")
  resultviolent = []
  resultproperty = []
  acsdata_df = readacs(acs)
  # Test Begin.

  # Test Segment
  tempresults = testiteration(2,testiternum,20000,25000,"None",0.135,0.01,False,acsdata_df, violentcsv, propertycsv, EPOCHS)
  resultviolent.append(tempresults[0])
  resultproperty.append(tempresults[1])
  # End Segment

  # Test Segment
  tempresults = testiteration(1,testiternum,20000,30000,"None",0.135,0.01,False,acsdata_df, violentcsv, propertycsv, EPOCHS)
  resultviolent.append(tempresults[0])
  resultproperty.append(tempresults[1])
  # End Segment

  # Test Segment
  tempresults = testiteration(3,testiternum,20000,40000,"None",0.135,0.01,False,acsdata_df, violentcsv, propertycsv, EPOCHS)
  resultviolent.append(tempresults[0])
  resultproperty.append(tempresults[1])
  # End Segment

  # Test Segment
  tempresults = testiteration(3,testiternum,50000,70000,"None",0.135,0.01,False,acsdata_df, violentcsv, propertycsv, EPOCHS)
  resultviolent.append(tempresults[0])
  resultproperty.append(tempresults[1])
  # End Segment

  # Test Segment
  tempresults = testiteration(3,testiternum,50000,100000,"None",0.135,0.01,False,acsdata_df, violentcsv, propertycsv, EPOCHS)
  resultviolent.append(tempresults[0])
  resultproperty.append(tempresults[1])
  # End Segment

  # Test Segment
  tempresults = testiteration(3,testiternum,10000,20000,"None",0.135,0.01,False,acsdata_df, violentcsv, propertycsv, EPOCHS)
  resultviolent.append(tempresults[0])
  resultproperty.append(tempresults[1])
  # End Segment

  # Test Segment
  tempresults = testiteration(3,testiternum,500,5000,"None",0.135,0.01,False,acsdata_df, violentcsv, propertycsv, EPOCHS)
  resultviolent.append(tempresults[0])
  resultproperty.append(tempresults[1])
  # End Segment

  # Test End
  printresults(resultviolent, resultproperty)

# VARIABLE IMPORTANCE
def test_B():
  print("Initializing testA()! This might take awhile.")
  resultviolent = []
  resultproperty = []
  acsdata_df = readacs(acs)
  # Test Begin.

  # Test Segment
  tempresults = testiteration(1,testiternum,20000,30000,"None",0.135,0.05,False,acsdata_df, violentcsv, propertycsv, EPOCHS)
  resultviolent.append(tempresults[0])
  resultproperty.append(tempresults[1])
  # End Segment

  # Test Segment
  tempresults = testiteration(1,testiternum,20000,30000,"None",0.135,0.03,False,acsdata_df, violentcsv, propertycsv, EPOCHS)
  resultviolent.append(tempresults[0])
  resultproperty.append(tempresults[1])
  # End Segment

  # Test Segment
  tempresults = testiteration(1,testiternum,20000,30000,"None",0.135,0.01,False,acsdata_df, violentcsv, propertycsv, EPOCHS)
  resultviolent.append(tempresults[0])
  resultproperty.append(tempresults[1])
  # End Segment

  # Test Segment
  tempresults = testiteration(1,testiternum,20000,30000,"None",0.135,0.007,False,acsdata_df, violentcsv, propertycsv, EPOCHS)
  resultviolent.append(tempresults[0])
  resultproperty.append(tempresults[1])
  # End Segment

  # Test Segment
  tempresults = testiteration(1,testiternum,20000,30000,"None",0.135,0.005,False,acsdata_df, violentcsv, propertycsv, EPOCHS)
  resultviolent.append(tempresults[0])
  resultproperty.append(tempresults[1])
  # End Segment

  # Test Segment
  tempresults = testiteration(1,testiternum,20000,30000,"None",0.135,0.003,False,acsdata_df, violentcsv, propertycsv, EPOCHS)
  resultviolent.append(tempresults[0])
  resultproperty.append(tempresults[1])
  # End Segment

  # Test Segment
  tempresults = testiteration(1,testiternum,20000,30000,"None",0.135,0.001,False,acsdata_df, violentcsv, propertycsv, EPOCHS)
  resultviolent.append(tempresults[0])
  resultproperty.append(tempresults[1])
  # End Segment
  
  # Test End
  printresults(resultviolent, resultproperty)

# TESTING STATE
def test_C():
  print("Initializing testA()! This might take awhile.")
  resultviolent = []
  resultproperty = []
  acsdata_df = readacs(acs)
  # Test Begin.

  # Test Segment
  tempresults = testiteration(1,testiternum,10000,30000,"None",0.135,0.01,False,acsdata_df, violentcsv, propertycsv, EPOCHS)
  resultviolent.append(tempresults[0])
  resultproperty.append(tempresults[1])
  # End Segment

  # Test Segment
  tempresults = testiteration(1,testiternum,10000,30000,"California",0.135,0.01,False,acsdata_df, violentcsv, propertycsv, EPOCHS)
  resultviolent.append(tempresults[0])
  resultproperty.append(tempresults[1])
  # End Segment

  # Test Segment
  tempresults = testiteration(1,testiternum,10000,30000,"Texas",0.135,0.01,False,acsdata_df, violentcsv, propertycsv, EPOCHS)
  resultviolent.append(tempresults[0])
  resultproperty.append(tempresults[1])
  # End Segment

  # Test End
  printresults(resultviolent, resultproperty)

# TEST
def test_X():
  print("Initializing testA()! This might take awhile.")
  resultviolent = []
  resultproperty = []
  acsdata_df = readacs(acs)
  # Test Begin.

  # Test Segment
  tempresults = testiteration(1,testiternum,50000,70000,"None",0.135,0.01,False,acsdata_df, violentcsv, propertycsv, EPOCHS)
  resultviolent.append(tempresults[0])
  resultproperty.append(tempresults[1])
  # End Segment

  # Test End
  printresults(resultviolent, resultproperty)

def printresults(resultviolent, resultproperty):
  print(seperationtext)
  print(" ALL DONE! RESULTS BEING PRINTED:")
  maeviolent = []
  maeproperty = []
  for item in resultviolent:
    maeviolent.append(item[1])
  for item in resultproperty:
    maeproperty.append(item[1])
  print("maeviolent:")
  print(maeviolent)
  print("maeproperty:")
  print(maeproperty)

def readacs(acs):
  print(" Reading in acs csv...")
  acsdata_df = pd.read_csv('acs/' + acs, index_col = 0) # acs csv file
  if acsdata_df is None:
      print('  ERROR! ' + str(acs) +' not found! Shutting down...')
      return
  print("  " + str(acs) + " read and parsed!")
  return acsdata_df

def testiteration(testnum, iternum, popthresh, popthreshupper, state, popmatchpercent,importancethresh,logbool,acsdataframebase, violentcsv, propertycsv, EPOCHS):
  results = [0,0]

  print(seperationtext)
  print("  RUNNING TEST NUMBER " +str(testnum)+ "...")
  print(seperationtext)
  autodatasetgenerator(iternum, popthresh, popthreshupper, state, popmatchpercent,importancethresh,logbool,acsdataframebase)
  results[0] = neuralnetwork(violentcsv, 'violent', EPOCHS)
  results[1] = neuralnetwork(propertycsv, 'property', EPOCHS)
  return results

if __name__ == "__main__":
  test_B()