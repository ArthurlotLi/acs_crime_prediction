# Takes in ucr_preprocessed.csv
# For each line in the csv, aka each data point, runs several functions:
#   Ensures no number in state and city name - if there is, removes it.
#   Discards all columns but two: Violent Crime and Property Crime.
#      Note that the program needs to check total number of rows to account for
#      the addition of the modified rape definition for 2013-2016 years.
#      14 for non-modified, 15 for modified. This will change where the
#      Violent crime and Property Crime target columns are.
#      Also note that if Violent/property crime columns are N/A, the
#      town is discarded.
# Outputs ucr.csv and ucrdiscarded.csv
# If -t is used, outputs all stdout into a log file called ucrlog.txt. 
#    Product csv should include: State, city, population, violent, property, and year. 

# Expects: ucrraw.csv
# Parameters: [-t] (optional)
# Outputs: ucr.csv 
#          ucrdiscarded.csv
#          ucrlog.txt*

import pandas as pd
import numpy as np
import argparse
import sys

rawcsvname = 'ucrraw.csv'
logname = 'ucrlog.txt'
productcsvname = 'ucr.csv'
discardedcsvname = 'ucrdiscarded.csv'

#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-t','--Text', required = False, action = "store_true")
args = parser.parse_args()

# The main function for this application, reading the initial
# .csv file. One that has been done, it initiates 
# functions that 1) discard N/A's as well as other irrelevant
# columns and 2) removes all numbers from stat and city names. 
# Finally, after all data preprocessing has been
# completed, writes ucr.csv out to the file. 
def processdocument():
    if args.Text:
        print("Outputting all system messages to ucrlog.txt!")
        file = open(logname, "w")
        sys.stdout = file

    print("ucrcompilation.py beginning to process csv...")
    # Read in csv into dataset format. (same directory)
    # Note that technically NaN is considered a float. All of these could be NaN.
    dataframe = pd.read_csv(rawcsvname, 
        dtype={"state":object,"city":object,"population":float,
        "violent":float,"property1":float,"property2":float,
        "year1":float,"year2":float})
    print("   Dataframe processed!")
    #print(dataframe)
    # Forwards dataset to discardempty which returns the dataset.
    resultdataframes = discardempty(dataframe)

    # Write into a new csv file and terminate. (same directory)
    productdataframe = resultdataframes['product']
    discardeddataframe = resultdataframes['discarded']
    print("   Writing product dataframes to files...")
    productdataframe.to_csv(productcsvname)
    print("      ucr.csv done!")
    discardeddataframe.to_csv(discardedcsvname)
    print("      ucrdiscarded.csv done!")
    if args.Text:
        file.close()
        sys.stdout = sys.__stdout__
    print("   Everything's done! Program exiting...")

# Accounts for modified rape definition and reduces all
# rows into just two number columns.
# discards rows with empty variables for crime/property.
def discardempty(dataframe):
    # Create a new dataset object 
    print("Formatting data...")
    data = {'state':[],'city':[],'population':[],'violent':[],'property':[],'year':[]}
    incompletedata = {'state':[],'city':[],'population':[],'violent':[],'property':[],'year':[]}
    
    # Only for pop-up notice
    previousstate = ""
    previousyear = ""
    firstrun = True

    for index, row in dataframe.iterrows(): # For each row
        stateval = row['state'] # stateval
        cityval = row['city'] # cityval
        populationval = row['population']# populationval
        violentval = row['violent'] # violentval
        # Year should never be nan, so convert it to an int. 
        if(np.isnan(row['year2'])): # If null, total of 14 columns.
            propertyval = row['property1']
            yearval = int(row['year1'])
        else:
            propertyval = row['property2']
            yearval = int(row['year2'])

        # Only for pop-up notice
        if(stateval != previousstate):
            if(firstrun):
                firstrun = False
            else:
                print("   Processed " + str(previousstate) + " for year " + str(previousyear) + "!")
            previousstate = stateval
            previousyear = yearval
        
        # For new dataset, if there are ANY n/a's, delete and announce in console.
        if(np.isnan(populationval) or np.isnan(violentval) 
            or np.isnan(propertyval)):
            # Unusable row. Remove and announce.
            print("      [Warning] Ignoring " + str(cityval) + ", " + str(stateval) + 
                ", population " + str(populationval) + " with violent " 
                + str(violentval) + " and property " + str(propertyval)) 
            incompletedata['state'].append(stateval)
            incompletedata['city'].append(cityval)
            incompletedata['population'].append(populationval)
            incompletedata['violent'].append(violentval)
            incompletedata['property'].append(propertyval)
            incompletedata['year'].append(yearval)
        else:
            # Usable row: no NaNs. Can safely turn into usable types. 
            # Removes numbers from state and city strings that were used for footnotes. 
            stateval = str(stateval)
            cityval = str(cityval)
            data['state'].append(''.join([i for i in stateval if not i.isdigit()]))
            data['city'].append(''.join([i for i in cityval if not i.isdigit()]))
            data['population'].append(int(populationval))
            data['violent'].append(int(violentval))
            data['property'].append(int(propertyval))
            data['year'].append(int(yearval))

    print("   All data processed. Fitting to dataframe...")
    incompletedataframe = pd.DataFrame(incompletedata)
    productdataframe = pd.DataFrame(data)
    print("      ...Done!")
    print("   Discarded Data:")
    print incompletedataframe
    print("   Final Dataset:")
    print productdataframe
    # Return new datasets.
    return {'product':productdataframe,'discarded':incompletedataframe}

if __name__ == "__main__":
    processdocument()