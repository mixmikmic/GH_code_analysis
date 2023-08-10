import pandas as pd
import math
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

class SlidingWindowDataCollection():
    """This is a base class for all algorithms that use a sliding window
    i.e. use n previous time steps to predict at the next time step"""
        
    def __init__(self, participants, sleep_info, test_info, demographic_features, output_info, pre_processing_info):
        """ Initialize Sliding Window Data Collection
        Parameters:
        ----------
        *participants: Participants object's participant_list attribute
        *sleep_info: dictionary {nPreviousNights: #, sleepFeatures: list}
        *test_info: dictionary {previousTestFeatures: list of lists [testName,testFeature], timingInfoPreviousTests: list, nPreviousTimesteps: #, timingInfoCurrentTest: list, ignoreFirstN: True/False}
        *demographic_features: list of demographic features (ones in subject information file)
        *output_info: dictionary {outputDataType: string, outputFeature: string, log_output: True/False}
        *pre_processing_info: dictionary {imputationType: string, normalize: True/False, columnsToNotNormalize: list}
        """
        
        self.participants = participants
        #sleep information
        self.sleep_info = sleep_info
        self.nPreviousNights = sleep_info['nPreviousNights']
        self.sleepFeatures = sleep_info['sleepFeatures']
        
        #test information
        self.test_info = test_info
        self.previousTestFeatures = test_info['previousTestFeatures']
        self.timingInfoPreviousTests = test_info['timingInfoPreviousTests']
        self.nPreviousTimesteps = test_info['nPreviousTimesteps']
        self.timingInfoCurrentTest = test_info['timingInfoCurrentTest']
        self.ignoreFirstN = test_info['ignoreFirstN']
        
        #demographic features
        self.demographicFeatures = demographic_features
        
        #output features
        self.output_info = output_info
        self.outputDataType = output_info['outputDataType']
        self.outputFeature = output_info['outputFeature']
        self.log_output = output_info['log_output']
        self.output_variable = self.outputDataType+self.outputFeature+"(t)"
        
        #pre-processing
        self.pre_processing_info = pre_processing_info
        self.normalize = pre_processing_info['normalize']
        self.columnsToNotNormalize = pre_processing_info['colsToNotNormalize']
        self.imputationType = pre_processing_info['imputationType']
        
        self.data = []
        self.column_names = []
        self.current_row = []
        self.current_names = []
        self.collectedDataFrame = None

    def getTimingFeature(self, feat, data_t, time, participant):
        """This function generates the timing features for the different tests
        
        Parameters:
        ----------
        *feat: feature name, options are 'HoursAwake', 'NumWPonFD', and 'CircadianPhase',
        NumWeekOnFD, SleepOpp72, HoursOnProtocol, SESSION, WP
        *data_t: dataframe of data at the time of the test we are considering
        *time: time of test we are considering
        *participant: Participant object we are collecting data for
        
        Returns:
        -------
        Value of the timing feature (float)
        
        """
        if len(data_t.index) == 0: #if we have no information for this time, return NaN
            d = np.nan
        elif feat == 'HoursAwake':
            d = data_t['HoursAwake'].values[0]
        elif feat == 'NumWPonFD':
            d = data_t['WakePeriod'].values[0]-participant.startFDSPn
        elif feat == 'NumWeekonFD':
            time = data_t['DecimalTime'].values[0]
            start_time = participant.startFDtime
            if time <= start_time+24*7:
                d = 1
            elif time <= start_time+24*7*2:
                d = 2
            elif time <= start_time+24*7*3:
                d = 3
            elif time <= start_time+24*7*4:
                d = 4
            else:
                assert 1==0, "issue with numWeek"
        elif feat == 'SleepOpp72':
            time_range = [data_t['DecimalTime'].values[0]-24*3,data_t['DecimalTime'].values[0]]
            timings = pd.read_csv("SleepTimingFile.csv")
            participant = participant.participantCode
            a = timings[timings.SUBJECT==participant]

            sleep_time = 0
            found = False
            for i in zip(a['Start'].values, a['End'].values):
                if time_range[0] >= i[0] and time_range[0] <= i[1]:
                    sleep_time += i[1]-time_range[0]
                    found = True
                    break
            if found:
                end_val_SP = i[1]
                remaining_time_range = [end_val_SP,time_range[1]]
            else:
                remaining_time_range = time_range
                                  
            for j in zip(a['Start'].values, a['End'].values):
                if j[0] >= remaining_time_range[0] and j[1] <= remaining_time_range[1]:
                    sleep_time += j[1]-j[0]
            return sleep_time
        elif feat == 'HoursOnProtocol':
            d = data_t['DecimalTime'].values[0]-participant.startFDtime
        elif feat == 'CircadianPhase':
            d = data_t['CircadianPhase'].values[0]
        elif feat == 'SESSION':
            d = data_t['SESSION'].values[0]
        elif feat == "WP":
            d = data_t['WakePeriod'].values[0]
        return d
    
    def getSleepData(self, participant, SPtoStart):
        """This function gets sleep metrics for a participant
        
        Parameters:
        ----------
        *participant: Participant object we are collecting data for
        *SPtoStart: number sleep period to start analysis at
        
        Returns:
        -------
        None
        """
        #get SPn numbers to consider, starting at SPtoStart
        sps_to_consider = [SPtoStart-i for i in range(self.nPreviousNights)] 
        d = participant.sleep

        for sp in sps_to_consider:
            for feat in self.sleepFeatures:
                d2 = d[d.SPn == sp][feat].values
                if len(d2) == 0:
                    self.current_row += [np.nan]
                elif len(d2) == 1:
                    self.current_row += [d2[0]]
                else:
                    assert 1==0, 'More than 1 row data for SP'
                self.current_names += ["PreviousSleepPeriod"+str(sps_to_consider.index(sp)+1)+feat]

    def getPreviousTestData(self, participant, t, hourStep):
        """Gets data for the previous n tests and test timing information
        The kth previous test (compared to the test at time t) is found by 
        looking at time t-k*hourStep. If the nearest test to that value is 
        within 30 minutes of it, then we use this test information. Otherwise
        we consider this test missing.
        
        Parameters:
        ----------
        *participant: Participant object we are collecting data for
        *t: time of the test we are trying to predict
        *hourStep: the time interval between the test batteries
        """
        names_to_data = {'ADD':participant.add, 'DSST':participant.dsst, 'PVT':participant.pvt, 'Moods':participant.moods}
        
        for testInfo in self.previousTestFeatures:
            testName = testInfo[0]
            testFeature = testInfo[1]
            data = names_to_data[testName]
            
            previousData = data[data.DecimalTime < t] #get all previous data

            for n in range(self.nPreviousTimesteps-1,-1,-1):
                c = previousData.iloc[(previousData['DecimalTime']-(t-hourStep*(n+1))).abs().argsort()[0:1]]

                if len(c.index) == 1:
                    closest_time =  c.DecimalTime.values[0]
                    if abs(closest_time-(t-hourStep*(n+1))) <= 0.5:
                        closest_time =  c.DecimalTime.values[0]
                        closest_score = (previousData[previousData.DecimalTime == closest_time][testFeature].values[0])
                    else:
                        closest_time = np.nan
                        closest_score = np.nan
                    time_diff = t-closest_time

                elif len(c.index) == 0:
                    closest_time = np.nan
                    closest_score = np.nan
                    time_diff = np.nan
                else:
                    assert 1==0,'More than one test at given time'+ str(t)

                self.current_row += [closest_score, closest_time, time_diff] 
                self.current_names += [str(testName)+str(testFeature)+"(t-"+str(n+1)+")",str(testName)+str(testFeature)+"(t-"+str(n+1)+")Time",str(testName)+str(testFeature)+"(t-"+str(n+1)+")TimeDiff"]

                #get test timing information
                for i in self.timingInfoPreviousTests:
                    self.current_row += [self.getTimingFeature(i, previousData[previousData.DecimalTime == closest_time], closest_time, participant)] 
                    self.current_names += [str(testName)+str(testFeature)+"(t-"+str(n+1)+")"+i]

    def collect_data(self):
        """This method constructs the data set for sliding window algorithms (only considers FD)
        
        Note: n previous tests aren't necessarily in the same WP (we only make sure that the output test
        has n previous tests in the same wake period if attribute ignoreFirstN = True in main driver function)
        """
        for participant in self.participants:
            names_to_data = {'ADD':participant.add, 'DSST':participant.dsst, 'PVT':participant.pvt, 'Moods':participant.moods}

            LabelData = names_to_data[self.outputDataType]
            LabelDataFD = LabelData[(LabelData.DecimalTime >= participant.startFDtime) & (LabelData.DecimalTime <= participant.endFDtime)]
            times = sorted(LabelDataFD.DecimalTime.values)
            dataset = []

            for t in range(len(times)):
                self.current_row = [times[t]]
                self.current_names = ['DecimalTime']
                data_t = LabelDataFD[LabelDataFD.DecimalTime == times[t]]
                label_t = data_t[self.outputFeature].values[0]
                wp = data_t['WakePeriod'].values[0]
                study = participant.study

                if self.ignoreFirstN: #check if the previous n output values are within same WP
                    data_prev = LabelDataFD[LabelDataFD.DecimalTime.isin([times[t-(i+1)] for i in range(self.nPreviousTimesteps)])]
                    valid_output = (len(set(data_prev['WakePeriod'].values)) == 1) & (data_t['WakePeriod'].values[0] == data_prev['WakePeriod'].values[0])
                else:
                    valid_output = True

                if valid_output and not np.isnan(label_t): #valid output, and make sure label is not missing
                    #get data for previous test features and test timing
                    if participant.study == 'AFOSR9':
                        self.getPreviousTestData(participant, times[t],4)
                    else:
                        self.getPreviousTestData(participant, times[t],2)

                    #get sleep data
                    self.getSleepData(participant, wp-1)

                    #get testTiming information
                    for i in self.timingInfoCurrentTest:
                        self.current_row += [self.getTimingFeature(i, data_t, times[t], participant)]
                        self.current_names += [i+'(t)']

                    #get demographic information
                    for i in self.demographicFeatures:
                        self.current_row += [getattr(participant, i)]
                        self.current_names += [i]

                    #get label
                    self.current_row += [label_t]
                    self.current_names += [self.outputDataType+self.outputFeature+"(t)"]

                    self.data.append(self.current_row)
                    self.column_names = self.current_names

        self.collectedDataFrame = pd.DataFrame(self.data,columns=self.column_names)
        return self.collectedDataFrame
    
    def getData(self):
        """Function that collects sliding window data, writes to file, and returns a dataframe
        If the data already exists, prompts user whether to overwrite.

        Returns:
        -------
        Returns dataframe and also writes data to file in Datasets folder
        """
        collect_data = False    
            
        fname = str(self.nPreviousTimesteps)+"previousSteps,"+'&'.join([str(i[1]) for i in self.previousTestFeatures]) + str(self.nPreviousNights)+"night"+str(len(self.sleepFeatures))+"->"+self.outputFeature
        if os.path.exists('Datasets/'+fname+'.csv'):
            response = input('Recompile this file '+fname+"?")
            if response == 'no':
                print("Using previous saved Data file",fname)
                collectedData = pd.read_csv('Datasets/'+fname+'.csv')
            elif response == 'yes':  #need to delete file
                collect_data = True
            else:
                print("Issue")
                collect_data = True
        else:
            print('Didnt find file, collecting new data')
            collect_data = True
            
        if collect_data:
            collectedData = self.collect_data()
            num_features = len(list(collectedData))-1
            
            collectedData.to_csv("Datasets/"+fname+'.csv', index=False)
            print("Collected Data for Filename"+fname)

            #write data information to txt file
            with open("Datasets/"+fname+'.txt', "w") as text_file:
                text_file.write("Sleep Info: "+str(self.sleep_info)+"\n") 
                text_file.write("Test Info: "+str(self.test_info)+"\n") 
                text_file.write("Demographic Info: "+str(self.demographicFeatures)+"\n") 
                text_file.write("Output Info: "+str(self.output_info)+"\n") 
                text_file.write("Pre-processing Info: "+str(self.pre_processing_info)+"\n") 
        return collectedData
    
    def noSplits(self, df):
        """ Returns all the data as both training, validation, and
        testing sets"""
        return [df], [df], [df]
    
    def predictSecondHalf(self, df):
        """ This implements Predict Second Half nested cross-validation 
        The original training and test split is done by splitting the wake
        periods in half. Then the validation set is the last three wake 
        periods of the training set.
        
        Parameters:
        ----------
        *df: dataframe of all data 
        
        Returns:
        -------
        *training_sets: list of dataframes to use for training data
        *testing_sets: list of dataframes to use for testing data
        *validation_sets: list of dataframes to use for validation data
        
        Ex. 
        Patient 1: [A1,A2,A3,A4,A5,A6,A7,A8,A9,A10]
        Patient 2: [B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12]
        
        Train/Test: train = [A1,B1-B2], validate = [A2-A4,B3-B5], test = [A5-A10,B6-B12]
        """
        training_data = []
        testing_data = []
        validation_data = []
        
        for participant in list(set(df['participantCode'])):
            df2 = df[df.participantCode == participant]
            df2 = df2.sort_values(by=['DecimalTime']) #make sure the data is ordered
            
            wakePeriods = sorted(list(set(df2['WP(t)'])))
            train_up_to = wakePeriods[int(len(wakePeriods)/2.0)]
        
            train = df[((df.participantCode == participant) & (df['WP(t)'] <= train_up_to-3))].reset_index(drop=True)
            validation = df[((df.participantCode == participant) & (df['WP(t)'] >= train_up_to-2)&(df['WP(t)'] <= train_up_to))].reset_index(drop=True)
            test = df[(df.participantCode == participant) & (df['WP(t)'] > train_up_to)].reset_index(drop=True)

            training_data.append(train)
            testing_data.append(test)
            validation_data.append(validation)
            
        try:
            training_sets = pd.concat(training_data).reset_index(drop=True)
            validation_sets = pd.concat(validation_data).reset_index(drop=True)
            testing_sets = pd.concat(testing_data).reset_index(drop=True)
            return [training_sets], [testing_sets], [validation_sets]
        except:
            return [training_data[0]], [testing_data[0]], [validation_data[0]]
    
    def populationInformedPredictSecondHalf(self,df):
        """ This implements population-informed Predict Second Half nested cross-validation 
        The code currently uses the last wake period in the training set
        as the validation set.
        
        Parameters:
        ----------
        *df: dataframe of all data 
        
        Returns:
        -------
        *training_sets: list of dataframes to use for training data
        *testing_sets: list of dataframes to use for testing data
        *validation_sets: list of dataframes to use for validation data
        
        Ex. 
        Patient 1: [A1,A2,A3,A4,A5,A6]
        Patient 2: [B1,B2,B3,B4,B5,B6]
        
        Train/Test #1: train = [A1-A2,B1-B6], validate = [A3], test = [A4-A6]
        Train/Test #2: train = [A1-A6,B1-B2], validate = [B3], test = [B4-B6]
        """
        
        training_data = []
        testing_data = []
        validation_data = []
        
        for participant in list(set(df['participantCode'])):
            df2 = df[df.participantCode == participant]
            df2 = df2.sort_values(by=['DecimalTime']) #make sure the data is ordered
            
            wakePeriods = sorted(list(set(df2['WP(t)'])))
            train_up_to = wakePeriods[int(len(wakePeriods)/2.0)]
        
            train = df[((df.participantCode == participant) & (df['WP(t)'] <= train_up_to-1)) | (df.participantCode != participant)].reset_index(drop=True)
            validation = df[((df.participantCode == participant) & (df['WP(t)'] == train_up_to))].reset_index(drop=True)
            test = df[(df.participantCode == participant) & (df['WP(t)'] > train_up_to)].reset_index(drop=True)

            training_data.append(train)
            testing_data.append(test)
            validation_data.append(validation)
        return training_data, testing_data, validation_data
    
    def populationInformedWPForwardChaining(self, df):
        """ This implements population-informed Wake Period
        Forward Chaining nested cross-validation. The code 
        currently uses the last wake period in the training set
        as the validation set.
        
        Parameters:
        ----------
        *df: dataframe of all data 
        
        Returns:
        -------
        *training_sets: list of dataframes to use for training data
        *testing_sets: list of dataframes to use for testing data
        *validation_sets: list of dataframes to use for validation data
        
        Ex. 
        Patient 1: [A1,A2,A3,A4,A5,A6]
        Patient 2: [B1,B2,B3,B4,B5,B6]
        
        Train/Test #1: train = [A1,B1-B5], validate = [A2], test = [A3]
        Train/Test #2: train = [A1,A2,B1-B5], validate = [A3], test = [A4]
        Train/Test #3: train = [A1,A2,A3,B1-B5], validate = [A4], test = [A5]
        Train/Test #4: train = [A1,A2,A3,A4,B1-B5], validate = [A5], test = [A6]
        Train/Test #5: train = [A1-A5,B1], validate = [B2], test = [B3]
        Train/Test #6: train = [A1-A5,B1,B2],validate = [B3], test = [B4]
        Train/Test #7: train = [A1-A5,B1,B2,B3], validate = [B4], test = [B5]
        Train/Test #8: train = [A1-A5,B1,B2,B3,B4], validate = [B5], test = [B6]
        """
        
        training_sets = []
        testing_sets = []
        validation_sets = []
        colnames = list(df)
        
        for participant in list(set(df['participantCode'])):
            df2 = df[df.participantCode == participant]
            df2 = df2.sort_values(by=['DecimalTime']) #make sure the data is ordered
            
            wakePeriods = sorted(list(set(df2['WP(t)'])))
            
            for wp in range(2,len(wakePeriods)):
                train = df[((df.participantCode == participant) & (df['WP(t)'] <= wakePeriods[wp-2])) | (df.participantCode != participant)].reset_index(drop=True)
                validation = df[((df.participantCode == participant) & (df['WP(t)'] == wakePeriods[wp-1]))].reset_index(drop=True)
                test = df[(df.participantCode == participant) & (df['WP(t)'] == wakePeriods[wp])].reset_index(drop=True)
            
                training_sets.append(train)
                testing_sets.append(test)
                validation_sets.append(validation)
        return training_sets, testing_sets, validation_sets
    
    def imputeTypes(self, values):
        """Calculates the value to impute with
        
        Parameters:
        ----------
        *values: values to impute
        
        Returns:
        --------
        Returns value to impute by
        """
        if self.imputationType == 'mean':
            return values.mean(axis=0)
        elif self.imputationType == 'median':
            return values.median(axis=0)
        elif self.imputationType == 'mode':
            return values.mode(axis=0)
        else:
            assert 1==0,'Imputation Type Not Implemented'
        
    def imputeAndNormalize(self, train, test, validate):
        """Function that imputes and normalizes data for each
        of the training/validation/testing splits. Imputation
        and normalization is performed column-wise based on the 
        values calculated from the training set.
        
        Parameters:
        -----------
        *train: training data to impute/normalize
        *test: test data to impute/normalize
        *validate: test data to impute/normalize
        
        Returns:
        -------
        Training and test data (now imputed and normalized as needed)
        """
        all_col = list(train)
        columnsToNormalize = [i for i in all_col if i not in self.columnsToNotNormalize]
        
        if self.imputationType == 'drop':
            train = train.dropna(axis=0, how='any')
            test = test.dropna(axis=0, how='any')
            validate = validate.dropna(axis=0, how='any')
        else:
            fillwith = self.imputeTypes(train)
            train = train.fillna(fillwith)
            test = test.fillna(fillwith)
            validate = validate.fillna(fillwith)
        
        for col in columnsToNormalize:
            test.loc[:,"Unnormalized:"+col] = test[col]
            train.loc[:,"Unnormalized:"+col] = train[col]
            validate.loc[:,"Unnormalized:"+col] = validate[col]
        
        means = train[columnsToNormalize].mean()
        stds = train[columnsToNormalize].std(ddof=0)    
        train[columnsToNormalize] = (train[columnsToNormalize]-means)/stds
        test[columnsToNormalize] = (test[columnsToNormalize]-means)/stds
        validate[columnsToNormalize] = (validate[columnsToNormalize]-means)/stds

        return train, test, validate
    
    def getSplitImputedNormalizedData(self, df, split_info):
        """Function that splits dataset, imputes, normalizes, and
        converts output variable, as necessary. The cross-validation
        splits are determined by the 'splitType' value of the split_info
        dictionary which can take the value of: 'none', 'predictSecondHalf',
        'populationInformedPredictSecondHalf', or 'populationInformedWPForwardChaining'
        
        Parameters:
        ----------
        *df: dataframe containing collected data
        *split_info: dictionary that contains preferences for splitting
        
        Returns:
        -------
        Imputed/Normalized training sets and Imputed/Normalized test sets
        """
        #split the datasets
        if split_info['splitType'] == 'populationInformedWPForwardChaining':
            training_sets, testing_sets, validation_sets = self.populationInformedWPForwardChaining(df)
        elif split_info['splitType'] == 'none':
            training_sets, testing_sets, validation_sets = self.noSplits(df)
        elif split_info['splitType'] == 'predictSecondHalf':
            training_sets, testing_sets, validation_sets = self.predictSecondHalf(df)
        elif split_info['splitType'] == 'populationInformedPredictSecondHalf':
            training_sets, testing_sets, validation_sets = self.populationInformedPredictSecondHalf(df)
        else:
            assert 1==0, "Split Type not Implemented"
        imputedNormalized_training_sets = []
        imputedNormalized_testing_sets = []
        imputedNormalized_validation_sets = []
        
        print("Splits made.....")
        all_col = list(df)
        columnsToNormalize = [i for i in all_col if i not in self.columnsToNotNormalize]
        print("Imputing and Normalizing Columns",columnsToNormalize)
        #Iterate through the splits and impute and normalize
        for i in range(len(training_sets)):
            train = training_sets[i]
            test = testing_sets[i]
            validate = validation_sets[i]
            
            imputedNormalized_trainingSet, imputedNormalized_testingSet, imputedNormalized_validationSet = self.imputeAndNormalize(train, test, validate)
            if self.log_output:
                imputedNormalized_trainingSet[self.output_variable] = np.log(imputedNormalized_trainingSet[self.output_variable])
                imputedNormalized_testingSet[self.output_variable] = np.log(imputedNormalized_testingSet[self.output_variable])
                imputedNormalized_validationSet[self.output_variable] = np.log(imputedNormalized_validationSet[self.output_variable])

                imputedNormalized_trainingSet.rename(columns={self.output_variable: 'Log:'+self.output_variable}, inplace=True)
                imputedNormalized_testingSet.rename(columns={self.output_variable: 'Log:'+self.output_variable}, inplace=True)
                imputedNormalized_validationSet.rename(columns={self.output_variable: 'Log:'+self.output_variable}, inplace=True)

            imputedNormalized_training_sets.append(imputedNormalized_trainingSet)
            imputedNormalized_testing_sets.append(imputedNormalized_testingSet)
            imputedNormalized_validation_sets.append(imputedNormalized_validationSet)
        
        return imputedNormalized_training_sets, imputedNormalized_testing_sets, imputedNormalized_validation_sets
    
    def split_data(self, collectedData, split_info):
        """ Function that splits the data into training, validation, and
        testing sets based on the 'splitType' variable in the split_info 
        dictionary. If split_info contains a 'studyChoice' variable, then
        only include participants from given study. If split_info contains
        a 'peopleChoice' variable, then only include specified individuals. 
        
        Parameters:
        ----------
        *collectedData: dataframe of collected data
        *split_info: dictionary of splitting information
        
        Returns:
        -------
        *training_sets: list of dataframes of training data, all imputed and normalized
        *validation_sets: list of dataframes of validation data, all imputed and normalized
        *testing_sets: list of dataframes of testing data, all imputed and normalized
        """
        
        if 'studyChoice' in split_info:
            collected_data_new = collectedData[collectedData.study.isin(split_info['studyChoice'])]
        elif 'peopleChoice' in split_info:
            collected_data_new = collectedData[collectedData.participantCode.isin(split_info['peopleChoice'])]   
        else:
            collected_data_new = collectedData

        training_sets, testing_sets, validation_sets = self.getSplitImputedNormalizedData(collected_data_new, split_info)
        return training_sets, testing_sets, validation_sets 


