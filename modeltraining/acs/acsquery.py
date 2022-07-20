# A program that takes in an iteration number 
# which correlates to the number in the name 
# of the list of variables and then queries 
# the ACS API for all those variables and then 
# compiles them into one csv file. 

# Note that csv files created are overwritten!

# Expects: acscrimeprediction/dataset/acs/acs_iter1_vars.txt
# Parameters: iter#, firstyear, lastyear (i.e. acsquery.py 1 2017 2018)
# Outputs: acscrimeprediction/dataset/acs/acs_iter1.csv

import pandas as pd
import numpy as np
import argparse
import sys
import requests
import json
import time

# Primary function that will handle the file input and output as
# well as hold the for loop that will repeatedly call the query
# function.

# Can also be called as a function with variables:
# firstyear
# lastyear
# inputname
# outputname
def generatecsv(firstyear, lastyear, inputname,outputname):
    # GLOBAL VARIABLES
    # Complete name EX) acs_iter1_vars.txt and acs_iter1.csv
    key = "" # TODO: Add key! 
    querypart1 = "https://api.census.gov/data/" # Insert year here. Ex) 2010
    querypart2 = "/acs/acs5/profile?get=" # Insert variables here. Ex) DP02_0017E
    querypart3 = "&for=place:*&in=state:*&key=" # Insert key here. 
    timeoutvar = 5000 # very long timeout for the API.
    population = 'DP05_0001E' # Population metric, for every query.

    wait = 2 # A "Niceness" wait time for the query after receiving a response. 
    variablelimit = 48 # How many variables can be queried from the api at once per query.
    #variablelimit = 1 # For testing.
    # NOTE: 49 is the definitive upper limit for the API. Don't try to query more at once; it won't let you.
    # END GLOBAL VARIABLES

    # Read in iter number. This is required. 
    print("Initializing generatcsv() for years " + str(firstyear) + " to " + str(lastyear) + " from input "+inputname+" to output "+outputname +".")
    #print("   Gotten iternum of " + str(iternum) + "!")

    # Given the iternum, look for txt file. If not there, throw error. 
    print(" Attempting to open " + inputname + "...")
    varsfile = open(inputname, "r")

    # Convert txt file contents into a string. If unable, throw error.
    vars = varsfile.read()
    if(vars == ''):
        print('  ERROR! NOTHING READ! Shutting down...')
        return
    varsfile.close()
    # Remove all instances of \r from the file... thanks excel.
    vars = vars.replace('\r','')
    populationinlist = False
    varslist = vars.split('\n')
    if population in varslist: # check if population exists in vars.
        print(' Population variable ' + population + ' found in vars!')
        numvars = len(varslist)+1 # Always +1 for NAME.
        populationinlist = True
    else:
        print(" Adding population variable " + population + ' to vars!')
        numvars = len(varslist) + 2 # Add 1 for population variable. 
        varslist.insert(0,population) # Add population to start of variables.
    print(' Total of ' + str(numvars) + ' variables! This will require ' + str(int(numvars/variablelimit+1)) + ' iterations per year.')

    # Get columns
    incompletecols = True
    cols = []

    masterdata = [] # all data will be a list of lists. 
    # Ex) [[u'Glenrock town, Wyoming', u'2498', u'1023', u'157', u'65', u'56', u'32435']]

    for year in range(firstyear,lastyear+1): # For all years 2010 to 2018...
        print("  Parsing year " + str(year) + "...")
        # For the list of all variables sectioned off into variablelimits.
        totaliterations = int(numvars/variablelimit+1)
        yeardata = None
        for i in range(totaliterations):
            print("  Calling acsquery (" + str(i+1) + " out of " + str(totaliterations) + ")")
            colsiteration = []
            varsiteration = varslist[(i*variablelimit):((i*variablelimit)+variablelimit)]
            if yeardata is None: # first run, add name variable.
                varsiteration.insert(0,'NAME')
                colsiteration.append('name')
            varsiterationcommas = ','.join(varsiteration)
            # Get columns
            for var in varsiteration:
                if incompletecols is True:
                    cols.append(str(var))
                colsiteration.append(str(var))
            if yeardata is None:
                if incompletecols is True:
                    cols.append('stateid')
                    cols.append('placeid')
                    cols.append('state')
                    cols.append('year')
                colsiteration.append('stateid')
                colsiteration.append('placeid')
                colsiteration.append('state')
                colsiteration.append('year')
            if(varsiterationcommas != ''):
                dataiteration = queryacs(year, varsiterationcommas, colsiteration, querypart1, querypart2, querypart3, key, timeoutvar, wait)
                if yeardata is None:
                    yeardata = dataiteration
                else: # Remove extraneous information if not first run.
                    for i in range(len(dataiteration)):
                        dataiteration[i]  = dataiteration[i][:len(dataiteration[i])-2] # Remove stateid, placeid.
                    for i in range(len(yeardata)):
                        for j in range(len(dataiteration[i])):
                            yeardata[i].append(dataiteration[i][j])
                print("  Iteration complete. Example yeardata is now " + str(yeardata[100]))
        #data = queryacs(year, vars, cols) # call helper funtion
        masterdata = masterdata + yeardata
        if incompletecols is True:
            print("  NOTE: COLS has been saved as: " + str(cols) + "!")
            incompletecols = False
    masterdataframe = pd.DataFrame(masterdata, columns=cols) # Consolidate all data into dataframe.

    print(" Writing results to "+outputname+"...")
    masterdataframe.to_csv(outputname, encoding='utf-8')
    print("...Done! Exiting program now! Goodbye!")

# Given a year, queries the spi.census.gov. Returns data
# that should be appended to the main dataset in the form
# of ????
def queryacs(year, vars, cols, querypart1, querypart2, querypart3, key, timeoutvar, wait):
    print("   Running queryacs for vars " + str(vars) + "...")
    query = querypart1 + str(year) + querypart2 + vars + querypart3 + key
    #print(query)
    request = None
    while request is None:
        try:
            request = requests.get(query, timeout=timeoutvar) # Query the program. Tiemout = 20
        except requests.exceptions.RequestException as e:
            print(e)
            print("    Query failed! Trying again...")
        if request is None:
            print("      ** Failed Query - Request = None! Wait... (Delay: "+ str(10)+") **")
            time.sleep(10)
            request = None
        elif request.text is None:
            print("      ** Failed Query - Request.text = None! Wait... (Delay: "+ str(10)+") **")
            time.sleep(10)
            request = None
        else:
            data = None
            try:
                data = json.loads(request.text)
            except:
                print("      ** Exception with json.loads!**")
            if data is None:
                print("      ** Failed Query - Data is none! Wait... (Delay: "+ str(10)+") **")
                time.sleep(10)
                request = None
            else:
                print("Success!")
        
    # If request is bad, try again. 
    data.pop(0) # Remove first element of list
    if 'NAME' in vars: # Only if name is in the variables.
        for row in data:
            placestate = row[0] # place, state.
            placestate = placestate.rpartition(',') # [0] = place, [2] = state
            row[0] = placestate[0] # Remove state from name.
            row.append(placestate[2]) # Add state col. 
            row.append(str(year))# format results and add year col. 
    if wait != 0:
        print("      ** Waiting... (Delay: "+ str(wait)+") **")
        time.sleep(wait)
    return data


if __name__ == "__main__":
    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('iternum')
    parser.add_argument('firstyear')
    parser.add_argument('lastyear')
    args = parser.parse_args()

    # Only if this function is being run by itself. 
    iternum = args.iternum
    firstyear = int(args.firstyear) # Year bounds.
    lastyear = int(args.lastyear)
    nameprefix = "acs_iter" 
    namesuffix = "_vars.txt"
    inputname = nameprefix + str(iternum) + namesuffix
    outputname = nameprefix + str(iternum) + ".csv"
    generatecsv(firstyear, lastyear, inputname, outputname)