import pandas as pd
import math
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
get_ipython().run_line_magic('matplotlib', 'inline')

class Participant():
    """This class contains all information for a given study participant
    as well as containing methods for plotting an individual's data"""
    
    def __init__(self, df, pvtTest, pvtDetailTest, addTest, dsstTest, moodsTest, sleepTest, sleepTiming):
        df = df.to_dict(orient = 'list')
        self.participantCode = df['SUBJECT'][0]
        self.study = df['STUDY'][0]
        self.age = int(df['Age'][0])
        self.gender = df['Gender'][0]
        
        #Information about study and Circadian cycle
        self.FDTCycle = float(df['FD T-cycle'][0])
        self.sleepLength = float(df['FD SP length'][0])
        self.wakeLength = float(df['FD WP Length'][0])
        self.startFDtime = float(df['Start analysis'][0])
        self.endFDtime = float(df['End Analysis'][0])
        self.startFDSPn = float(df['Start analysis SPn'][0])
        self.endFDSPn = float(df['End analysis SPn (included)'][0])
        
        #store data
        self.pvt = pvtTest[pvtTest.SUBJECT == self.participantCode]
        self.pvtDetail = pvtDetailTest[pvtDetailTest.SUBJECT == self.participantCode]
        self.FDsleepPeriods = sleepTiming[sleepTiming.SUBJECT==self.participantCode][['SP','Start','End']].values
        self.add = addTest[addTest.SUBJECT == self.participantCode]
        self.dsst = dsstTest[dsstTest.SUBJECT == self.participantCode]
        self.moods = moodsTest[moodsTest.SUBJECT == self.participantCode]
        self.sleep = sleepTest[sleepTest.SUBJECT == self.participantCode]
        
    def mask(self, y, times):
        """Helper function for plotting which adds in empty areas during the
        sleep periods (using information from FDsleepPeriods)
        
        Parameters:
        ----------
        y: y values
        t: time values
        
        Returns:
        -------
        new_t: time values which are now masked
        y_values_masked: y values which are now masked
        """
        timeVal = [[times[i],y[i]] for i in range(len(times))]
        starts = [i[1] for i in self.FDsleepPeriods]
        ends = [i[2] for i in self.FDsleepPeriods]
        
        #adds in synthetic points during sleep periods
        for i in range(len(starts)):
            for j in np.arange(starts[i], ends[i], 0.01):
                timeVal.append([j,-999])

        timeVal = sorted(timeVal)
        new_y = [i[1] for i in timeVal]
        new_t = [i[0] for i in timeVal]

        y_values = np.ma.array(new_y)
        y_values_masked = np.ma.masked_where(y_values == -999 , y_values)
        return new_t, y_values_masked
    
    def plotRawWithSleepPeriods(self, title=None, scatter=True):
        """
        This function plots the raw Mean Inverse RT on the PVT
        test within the Forced Desynchrony period
        
        Parameters:
        ----------
        title: title for the plot
        scatter: whether to include the scatter points
        """
        if not title:
            title = str('PVT Data for Participant '+self.participantCode)
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111)
        data = self.pvt

        data = data[(data.DecimalTime >= self.startFDtime) & (data.DecimalTime <= self.endFDtime)]
        data = data.sort_values(by=['DecimalTime'])

        y = list(data['MeanInverseRT'])
        times = list(data['DecimalTime'])
        min_time = min(times)
        new_times, y_values_masked = self.mask(y, times)
        zeroed_times = new_times-min_time

        starts = [i[1]-min_time for i in self.FDsleepPeriods]
        ends = [i[2]-min_time for i in self.FDsleepPeriods]
        middleOfWakePeriods = [(starts[i+1]+ends[i])/2.0 for i in range(len(starts)-1)]
        middleOfFDWakePeriods = [i for i in middleOfWakePeriods if i >= 0 and i <= self.endFDtime-min_time]

        ax.plot(zeroed_times, y_values_masked)
        ax.scatter(zeroed_times, y_values_masked)

        ax.set_ylim([0,5.5])
        ylim = [0,5.5]
        for i in range(len(starts)):
            if starts[i] >= self.startFDtime-min_time and starts[i] <= self.endFDtime-min_time:
                ax.add_patch(
                    patches.Rectangle(
                        (starts[i], ylim[0]),   # (x,y)
                        ends[i]-starts[i],          # width
                        ylim[1]-ylim[0],          # height
                        alpha = 0.1,
                        color = 'gray'
                    )
                )

        ax.set_xlim(times[0]-5-min_time,times[-1]+5-min_time)
        ax.set_xlabel('Wake Period on Forced Desychrony Protocol', fontsize=18)
        ax.set_ylabel(str("PVT Mean Response Speed (1/sec) "), fontsize=18)
        ax.set_title(title, fontsize=18)
        ax.set_xticks(middleOfFDWakePeriods)
        ax.set_xticklabels([i+1 for i in range(len(middleOfFDWakePeriods))])
        ax.tick_params(labelsize=15)
        plt.show()
        
    def plot_results(self, times, y, prediction, title=None):
        """This function plots the true data versus the predictions
        
        Parameters:
        ----------
        times: the times corresponding to the y values
        y: the true output values
        prediction: the predicted output values
        """
        if not title:
            title = str('PVT Predicted vs. Actual Data for Participant '+self.participantCode)
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111)
        
        assert len(y) == len(times), "length of y and times not equal"
        assert len(prediction) == len(times), "length of y and times not equal"
        
        times = list(times)
        y = list(y)
        prediction = list(prediction)
        min_time = min(list(times))
        new_times, y_values_masked = self.mask(y, times)
        zeroed_times = new_times-min_time
        
        new_times_pred, y_values_masked_pred = self.mask(prediction, times)
        zeroed_times_pred = new_times_pred-min_time

        starts = [i[1]-min_time for i in self.FDsleepPeriods]
        ends = [i[2]-min_time for i in self.FDsleepPeriods]
        middleOfWakePeriods = [(starts[i+1]+ends[i])/2.0 for i in range(len(starts)-1)]
        middleOfFDWakePeriods = [i for i in middleOfWakePeriods if i >= 0 and i <= self.endFDtime-min_time]

        ax.plot(zeroed_times, y_values_masked, 'g', label='Observed Data')
        ax.scatter(zeroed_times, y_values_masked, c='g')
        ax.plot(zeroed_times_pred, y_values_masked_pred, 'r', label='Predicted Data')
        ax.scatter(zeroed_times_pred, y_values_masked_pred, c='r')

        ax.set_ylim([0,5.5])
        ylim = [0,5.5]
        for i in range(len(starts)):
            if starts[i] >= self.startFDtime-min_time and starts[i] <= self.endFDtime-min_time:
                ax.add_patch(
                    patches.Rectangle(
                        (starts[i], ylim[0]),   # (x,y)
                        ends[i]-starts[i],          # width
                        ylim[1]-ylim[0],          # height
                        alpha = 0.1,
                        color = 'gray'
                    )
                )

        ax.set_xlim(times[0]-5-min_time,times[-1]+5-min_time)
        ax.set_xlabel('Wake Period on Forced Desychrony Protocol', fontsize=18)
        ax.set_ylabel(str("PVT Mean Response Speed (1/sec) "), fontsize=18)
        ax.set_title(title, fontsize=18)
        ax.set_xticks(middleOfFDWakePeriods)
        ax.set_xticklabels([i+1 for i in range(len(middleOfFDWakePeriods))])
        ax.tick_params(labelsize=15)
        plt.show()
        

class Participants():
    """This class contains information for all the paricipants in our data set"""
    
    def __init__(self):
        participants = pd.read_csv("SubjectInformation.csv", na_values = ['','.'],encoding="latin-1")
        pvtTest = pd.read_csv("PVTSummaryData.csv", na_values = ['','.'], low_memory = False,encoding="latin-1")
        pvtDetailTest = pd.read_csv("PVTRawData.csv", na_values = ['','.'], low_memory = False,encoding="latin-1")
        sleepTiming = pd.read_csv("SleepTimingFile.csv", na_values = ['','.'], low_memory = False,encoding="latin-1")
        sleepTest = pd.read_csv("SleepInformation.csv", na_values = ['','.'], low_memory = False,encoding="latin-1")
        #placeholders for real files
        addTest = pd.DataFrame({'SUBJECT':[''],'CORRECT':[np.nan]})
        dsstTest = pd.DataFrame({'SUBJECT':[''],'CORRECT':[np.nan]})
        moodsTest = pd.DataFrame({'SUBJECT':[''],'ALERT':[np.nan]})
        
        self.participant_list = []
        self.participant_dict ={}
        for i in list(set(participants.SUBJECT)):
            p = Participant(participants[participants.SUBJECT == i],  pvtTest, pvtDetailTest, addTest, dsstTest, moodsTest, sleepTest, sleepTiming)
            self.participant_dict[i] = p
            self.participant_list.append(p)
            
    def plot_results(self, resultsDF):
        for p in list(set(resultsDF.participantCode)):
            df = resultsDF[resultsDF.participantCode == self.participant_dict[p].participantCode]
            self.participant_dict[p].plot_results(times = df.DecimalTime.values, y = df['PVTMeanInverseRT(t)'].values, prediction =df['prediction'].values)
        

