# 
# autodataset.py
#
# Master dataset creation tool that allows for automated feature 
# selection. Utilize this class to generate a dataset with
# specific qualities to be used in conjuncion with a model
# training class. 
#
# Executes the following steps during runtime:
#  1. Queries the American Community Survey API for data to
#     populate the acscomplete.csv (a giant csv with ALL ACS
#     data from 2010 to 2019.) If the file exists, this step
#     will be skipped. 
#  2. Merges the ACS data with UCR data, correlating cities
#     in the ACS entries with police department entries 
#     while referencing substrings and population differences. 
#     The popmatchpercent value is utilized to address 
#     discrepancies.
#  3. Executes supervised feature selection provided a
#     specified method (refer to complete/feature_reduction.py
#     for supported methods). 
#  4. Saves the resulting dataset with the specified iteration
#     number to a csv, ready for use in a machine learning 
#     model training procedure.

# Uses the following files:
#  - acs/acsquery.py
#  - feature_reduction.py

# Also required are the following documents:
#  - ucr/ucr.csv
#  - acs/acsvariablesmain.txt

# We will generate the following file if it is 
# not found (recommend you get this from someone 
# rather than querying though.)
#  - acs/acscomplete.csv

# Will output the following file based on
# the iternum variable:
#  - complete/complete_iterA#.csv!

# Variables required: iternum, popthresh, popthreshupper, state, popmatchpercent, feature_selection_method
#  - Ex) python autodataset.py 1 10000 100000 none 0.17 pearson_selectkbest
#
# NOTE: If you want to save the console output (especially for when 
# you're generating acscomplete.csv), add -t to write to log.
#  - Ex) python autodataset.py -t 1 10000 50000 none 0.10 rfe_tree

import pandas as pd
import argparse
import sys
from os import path

# Grab the functions needed from aforementioned files. 
from acs.acsquery import generatecsv # For generating acscomplete.csv if necessary.
#from variablereduction import reducevariables <-- Obsolete method using RF Regressor.
#from dimensionality_reduction import DimensionalityReduction <-- Transform features, which is undesirable for desired model transparency
from feature_reduction import FeatureReduction

class AutoDataset:
    # Change only for debugging (i.e. acs_iter2), otherwise should stay as acscomplete.
    #databasename = "acs_iter3"
    databasename = "acscomplete"
    logname = "autodataset_log.txt"
    seperationtext = "=================================================================================="

    def autodatasetgenerator(self, iternum, popthresh, popthreshupper, state, popmatchpercent,feature_reduction_method, logbool,acsdataframebase = None):
        # Before everything else, direct output to log file if directed to do so
        if logbool:
            print("Outputting all system messages to "+self.logname+"!")
            file = open(self.logname, "w")
            sys.stdout = file
        print(self.seperationtext + "\n[INFO] Initializing autodataset() for iteration " + str(iternum) + "...")

        reducer = FeatureReduction()
        if reducer.check_method_present(feature_reduction_method) is not True:
            print("[ERROR] Feature reduction method '"+feature_reduction_method+"' does not exist. Exiting!")
            return
        
        # If acsdataframebase given, do not look for file. 
        if acsdataframebase is None:
            # Check if acscomplete.csv exists in acs/acscomplete.csv.
            if path.exists("acs/" + self.databasename+".csv"):
                print(" Found " +self.databasename+".csv! Yay!")
            else: # If no, run generatecsv, 2010, 2018, 'acsvariablesmain.txt', 'acscomplete.csv' 
                # NOTE: 2019 ACS variables have (annoyingly) been transformed compared to 2010-2018. 
                # for the sake of simplicity, we'll be using only 2010 to 2018 for now. 
                if path.exists("acs/" +self.databasename+ ".txt"):
                    print(" Generating "+self.databasename+".csv... this may take awhile...")
                    print(self.seperationtext)
                    generatecsv(2010, 2018, ("acs/" + self.databasename+'.txt'), ("acs/" + self.databasename+".csv"))
                    print(self.seperationtext)
                else:
                    print(" ERROR! "+self.databasename+".txt does not exist! Exiting!")
                    return
        # We now expect acscomplete.csv to exist. 

        # Run createcomplete with acscomplete.csv. 
        print("[INFO] Applying variables and merging with UCR data...")
        print(self.seperationtext)
        unpruneddata = self.createcompletedataset(self.databasename+".csv",popthresh,popthreshupper,state,popmatchpercent,acsdataframebase)
        print(self.seperationtext)
        print("[INFO] UCR merge complete.")

        # Given that dataframe, run reducevariables with property and violent. 
        productprefix = 'complete_iterA'
        property_productsuffix = '_property.csv'
        violent_productsuffix = '_violent.csv'
        property_file_path = productprefix + str(iternum) + property_productsuffix
        violent_file_path = productprefix + str(iternum) + violent_productsuffix
        
        print("[INFO] Reducing variables for property...\n" + self.seperationtext)
        productdataframe = reducer.reduce_dataset(unpruneddata,'property', feature_reduction_method)
        # RF Regressor method (obsolete): productdataframe = reducevariables(unpruneddata,importancethresh,'property')
        print(self.seperationtext + "\n[INFO] Writing results to "+property_file_path + "...")
        productdataframe.to_csv("complete/"+ property_file_path, index=False)

        print("[INFO] Reducing variables for violent...\n" + self.seperationtext)
        productdataframe = reducer.reduce_dataset(unpruneddata,'violent', feature_reduction_method)
        # RF Regressor method (obsolete): productdataframe = reducevariables(unpruneddata,importancethresh,'violent')
        print(self.seperationtext + "\n[INFO] Writing results to "+violent_file_path + "...")
        productdataframe.to_csv("complete/"+violent_file_path, index=False)

        # Write a final README to note the qualitities of the dataset (in case we forget).
        readme = productprefix + str(iternum) + ".txt"
        print("[INFO] Writing note to readme " + readme + "...")
        readmefile = open("complete/"+readme, "w")
        readmefile.write(str(iternum) +  " " + str(popthresh) + " " + str(popthreshupper) + " " + str(state) + " "  + str( popmatchpercent))
        readmefile.close()

        # Close the log if we're loggin'. 
        if logbool:
            file.close()
            sys.stdout = sys.__stdout__

        print(" ...And we're all done! Have a nice day! :)")
        # Done. Exit. 
        return

    #
    # NOTE: The following code was formerly part of the createcomplete.py 
    # file, now merged into autodataset. 
    #

    # A program that, given a large amount of 
    # ACS data and the base UCR data, combines 
    # them together into a final dataset, 
    # correlating places + states with UCR 
    # data points and discarding places in 
    # the ACS dataset without matching UCR 
    # departments. 

    # Expects: acs/acs_iter1.csv
    #	       ucr/ucr.csv
    # Parameters: iter# popthresh state popmatchpercent (i.e. createcomplete.py 1 50000 California 0.10)
    # NOTE: State can simply be 'None', in which case all states are parsed from ACS.
    # Outputs: complete/complete_iter1.csv


    # What percent of population can deviate?!
    # EX) ACS Los Angeles = 3772486. UCR Los Angeles = 4029741. Difference: 257255, or 0.063 difference.

    # Primary function that deals with I/O, etc. 
    # Calls mergedatasets.

    # Can be used as a function that outputs a productdataframe as well.

    # inputname (expects a csv, most likely acscomplete.csv if using autodataset.py)
    # popthresh
    # state
    # popmatchpercent
    def createcompletedataset(self, inputname, popthresh, popthreshupper, state, popmatchpercent,acsdataframebase):
        # GLOBAL VARIABLES:
        ucr = 'ucr.csv'
        # END GLOBAL VARIABLES

        if(popthresh >= popthreshupper):
            print("  ERROR! popthresh >= popthreshupper! Shutting down...")
            return

        acs = inputname
        print("Initializing generatcsv()!")

        print(" Attempting to read csv files...")
        # read in both csv files, in quotes, put in path or file name for csv file with data
        if acsdataframebase is None:
            acsdata_df = pd.read_csv('acs/' + acs, index_col = 0) # acs csv file
            if acsdata_df is None:
                print('  ERROR! ' + str(acs) +' not found! Shutting down...')
                return
            print("  " + str(acs) + " read and parsed!")
        else:
            acsdata_df = acsdataframebase
            print("  acsdataframebase passed in! No need to read!")
        ucrdata_df2 = pd.read_csv('ucr/' + ucr, index_col = 0) # ucr csv file
        if ucrdata_df2 is None:
            print('  ERROR! ' + str(ucr) +' not found! Shutting down...')
            return
        print("  " + str(ucr) + " read and parsed!")
        productdataframe = self.mergeucracs(acsdata_df, ucrdata_df2, popthresh, popthreshupper, state, popmatchpercent)
        return productdataframe

    # mergeucracs merges the two dataframes based on popthresh and state. 
    # Methodology is the following: 
    #    given a city in the ucr dataset, find the city name substring in
    #    all of the entries in acs with that same state and year variables.

    #    Once that has been done, find the candidate with the closest population
    #    varaible (DP05_0001E) to the population of the ucr data. 

    #    If it is within a reasonable degree (this is a global percentage varaible!!)
    #    then accept it, and add it to the list to be made into a final dataframe.
    #    If it is not, there are no acceptable candidates - reject and alert user.
    def mergeucracs(self, acsdata_df, ucrdata_df2, popthresh, popthreshupper, stateparam, popmatchpercent):
        print(" Merging datasets for popthresh " + str(popthresh) + " and popthreshupper " +str(popthreshupper)+" and state " + str(stateparam) + ".")

        print("  Getting column names...")
        # Grab columns.
        merged = pd.merge(acsdata_df, ucrdata_df2, left_on=['NAME'], right_on=['city'], how='inner') 
        cols = merged.columns.values.tolist()
        print("   Columns names found! They are: " + str(cols))

        masterlist = []
        totalmissed = 0
        partialmisses = 0
        successes = 0

        # Keep track of how many times we've seen each city. 
        cityrowshashtable = {}

        # All ucr cities.
        ucrkeys = ucrdata_df2.index.tolist()
        ucrcities = ucrdata_df2['city'].tolist()
        ucrcities = list(map(lambda x:x.lower().strip(),ucrcities))
        ucryears = ucrdata_df2['year'].tolist()
        ucrstates = ucrdata_df2['state'].tolist()
        ucrstates = list(map(lambda x:x.lower().strip(),ucrstates))
        ucrpops = ucrdata_df2['population'].tolist()

        # All ACS places.
        acskeys = acsdata_df.index.tolist()
        acsplaces = acsdata_df['NAME'].tolist()
        acsplaces = list(map(lambda x:x.lower().strip(),acsplaces))
        acsyears = acsdata_df['year'].tolist()
        acsstates = acsdata_df['state'].tolist()
        acsstates = list(map(lambda x:x.lower().strip(),acsstates))
        acspops = acsdata_df['DP05_0001E'].tolist()

        # processing speed increase with combination of blacklists and whitelists.
        yearblacklist = [2019] # NOTE: 2019 ACS variables have been transformed. Do not use for now. 
        yearwhitelist = []
        stateblacklist = []
        statewhitelist = []

        for i in range(len(ucrcities)):
            year = ucryears[i]
            if year not in yearblacklist: # Not blacklisted
                if year not in yearwhitelist: # Not confirmed to be good.
                    if year not in acsyears: # Confirm it is bad
                        yearblacklist.append(year)
                        print("  YEAR BLACKLIST - Added " + str(year) + " to blacklist.")
                        continue
                    else: # Confirm it is good.
                        yearwhitelist.append(year)
                        print("  YEAR WHITELIST - Added " + str(year) + " to whitelist!")
            else: # Blacklisted year.
                continue

            state = ucrstates[i]
            if stateparam.lower() != 'none':
                if state != stateparam.lower():
                    continue
            if state not in stateblacklist: # Not blacklisted
                if state not in statewhitelist: # Not confirmed to be good.
                    if state not in acsstates:  # Confirm it is bad.
                        stateblacklist.append(state)
                        print("  STATE BLACKLIST - Added " + state + " to blacklist.")
                        continue
                    else: # Confirm it is good.
                        statewhitelist.append(state)
                        print("  STATE WHITELIST - Added " + state + " to whitelist!")
            else: # Blacklisted state.
                continue

            ucrpop = int(ucrpops[i])
            # Check population threshold. If too low or too high, skip.
            if ucrpop < popthresh:
                continue
            if ucrpop > popthreshupper:
                continue

            # Find all rows in the acs data with a NAME that contains 'city' as a substring. 
            # Ex) UCR entry "Los Angeles" inside ACS entry "Los Angeles City" and "East Los Angeles City"
            city = ucrcities[i]
            candidatespops = []
            candidatesindexs = []
            for j in range(len(acsplaces)):
                if acsyears[j] == year:
                    if acsstates[j].lower() == state:
                        if city in acsplaces[j].lower():
                            candidatesindexs.append(j)
                            candidatespops.append(int(acspops[j]))

            # All candidates gathered. Find index with pop value closest to ucrpop.
            popdiff = 2147483646 # essentially inifity -> 2 billion.
            closest = None
            for k in range(len(candidatespops)):
                candidatediff = abs(candidatespops[k] - ucrpop)
                if candidatediff < popdiff:
                    popdiff = candidatediff
                    closest = candidatesindexs[k]
            
            subject = ucrdata_df2.loc[ucrkeys[i]]
            subjectrow = subject.tolist()
            if closest is not None:
                candidate = acsdata_df.loc[acskeys[closest]]
                candidaterow = candidate.tolist()
                if popdiff < (ucrpop*popmatchpercent):
                    # Accepted!
                    productrow = [] 
                    for k in range(len(candidaterow)):
                        productrow.append(candidaterow[k])
                    for k in range(len(subjectrow)):
                        productrow.append(subjectrow[k])
                    masterlist.append(productrow)
                    # Update hash table
                    indexplaceid = cols.index('placeid')
                    if indexplaceid is not None:
                        placeid = productrow[indexplaceid]
                        if placeid not in cityrowshashtable:
                            cityrowshashtable[placeid] = 1
                        else:
                            cityrowshashtable[placeid] = cityrowshashtable[placeid] + 1
                        #print("  ...Merged " + str(subject['year']) + " "+ str(subject['city']) + ", " + str(subject['state']) + " with pop " + str(subject['population']) + " with " + str(candidate['NAME']) + " with pop " + str(candidate['DP05_0001E']) + "!")
                        successes = successes+1
                else:
                    print("!!!WARNING - Closest candiate for " + str(subject['year']) + " "+ str(subject['city']) + ", " + str(subject['state']) + " with pop " + str(subject['population']) + " was: " + str(candidate['NAME']) + " with pop " + str(candidate['DP05_0001E']) + ", with a popdiff of " + str(popdiff) + "(" + str(float(popdiff)/ucrpop) + ")"  + " out of " + str(ucrpop*popmatchpercent) + "(" + str(popmatchpercent) +")"+ ".")
                    partialmisses = partialmisses+1
                    totalmissed = totalmissed+1
            else:
                print("!!!WARNING - No candidate found for " + str(subject['year']) + " "+ str(subject['city']) + ", " + str(subject['state']) + " with pop " + str(subject['population']) + "!")
                totalmissed = totalmissed+1

        # merges both datasets based on the name of the place, gets rid of any datapoints where the place names do not match
        #mergedacsucr_df = pd.merge(acsdata_df, ucrdata_df2, left_on=['NAME'], right_on=['city'], how='inner') 

        # filters out datapoints whose population size is less than the threshold value set
        #lowerthanThreshold_acs = mergedacsucr_df[ mergedacsucr_df['Population_x'] < popthresh].index

        # commented out, but could also be used to filter out dataset, does the same thing as the command above
        # lowerthanThreshold_ucr = mergedacsucr_df[ mergedacsucr_df['Population_y'] < threshold].index

        # drop the rows with a population less than given threshold
        #mergedacsucr_df.drop(lowerthanThreshold_acs, inplace=True)

        # commented out, but drops rows w population less than given threshold. 
        # mergedacsucr_df.drop(lowerthanThreshold_ucr, inplace=True)
        # print(mergedacsucr_df) - testing purposes LOL

        # Ignore towns that do not have entries for all years by checking the hash list value.
        totalyears = len(yearwhitelist)
        finallist = []
        insufficientYears = 0

        print("  Total number of years is " + str(totalyears) + ". masterlist contains " + str(len(masterlist)) + " rows. Checking hash tables...")
        indexplaceid = cols.index('placeid')
        indexname = cols.index('NAME')
        if indexplaceid is not None:
            for row in masterlist:
                if row[indexplaceid] in cityrowshashtable:
                    if cityrowshashtable[row[indexplaceid]] == totalyears:
                        finallist.append(row)
                    else:
                        insufficientYears = insufficientYears+1
                        if indexname is not None:
                            print("!!!WARNING - dropping " + str(row[indexname]) + "; only " + str(cityrowshashtable[row[indexplaceid]]) + " row entries out of " + str(totalyears) + " years.")
                else:
                    insufficientYears = insufficientYears+1
                    if indexname is not None:
                            print("!!!WARNING - dropping " + str(row[indexname]) + "; 0 row entries out of " + str(totalyears) + " years.")

        else:
            finallist = masterlist


        masterdataframe = pd.DataFrame(finallist, columns=cols) # Consolidate all data into dataframe.

        print("  Parse complete. Total missed: " + str(totalmissed) + ", Partial Misses: " + str(partialmisses) + ". Insufficient Years: " + str(insufficientYears) + ". Total Successful: " + str(successes - insufficientYears) + "!")

        return masterdataframe

if __name__ == "__main__":
    iternum = None
    popthresh = None
    popthreshupper = None
    state = None
    popmatchpercent = None
    feature_reduction_method = None
    text = None
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-t','--Text', required = False, action = "store_true")
        parser.add_argument('iternum')
        parser.add_argument('popthresh')
        parser.add_argument('popthreshupper')
        parser.add_argument('state')
        parser.add_argument('popmatchpercent')
        parser.add_argument('feature_reduction_method')
        args = parser.parse_args()

        iternum = args.iternum
        popthresh = int(args.popthresh)
        popthreshupper = int(args.popthreshupper)
        state = args.state
        popmatchpercent = float(args.popmatchpercent)
        feature_reduction_method = args.feature_reduction_method
        text = args.Text
    except:
        # Running in debug.     
        iternum = 1
        popthresh = 50000
        popthreshupper = 60000
        state = "none"
        popmatchpercent = 0.17
        feature_reduction_method = "pearson_selectkbest"
        text = None

    auto_dataset = AutoDataset()
    auto_dataset.autodatasetgenerator(iternum,popthresh,popthreshupper,state,popmatchpercent, feature_reduction_method, text, None)