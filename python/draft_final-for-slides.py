from make_df import *
import nltk
from nltk.collocations import *

import pandas as pd
import string
from Trend import Trend
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from make_years import *
import numpy as np

from make_bigrams import *

from gensim.models import Word2Vec

import pickle
from collections import Counter


df1 = pd.read_pickle('/dataxvdf/masterA-1128.pkl')
#df1 = get_columns_for_nlp(df1)

df2 = pd.read_pickle('/dataxvdf/masterB-1128.pkl')

df3 = pd.read_pickle('/dataxvdf/masterC-1128.pkl')

dfX = pd.concat([df1, df2, df3])

mask = dfX.post_id != 0

dfY = dfX[mask]

dfY.describe()

def make_year(df, year):
    mask = df.year == year
    return df[mask]

def make_quarter(df, year, quarter):
    year_df = make_year(df, year)
    mask = year_df.quarter == quarter
    return year_df[mask]

df2008 = make_year(dfY, 2008)
df2009 = make_year(dfY, 2009)
df2010 = make_year(dfY, 2010)
df2011 = make_year(dfY, 2011)
df2012 = make_year(dfY, 2012)
df2013 = make_year(dfY, 2013)
df2014 = make_year(dfY, 2014)
df2015 = make_year(dfY, 2015)
df2016 = make_year(dfY, 2016)

df2008 = df2008[df2008.datetime > '2008-03-01']

df2016.describe()

#this df will be used for plotting over time. Right now I am looking at 2009-2013.
#2014-2016 data has been set aside for validation
dfZ = pd.concat([df2009, df2010, df2011, df2012, df2013, df2014, df2015, df2016])

model = Word2Vec.load('w2v.txt')

basic_garments = ["dress", "pants", "shoes", "shirt", "bag"]

garment_list = []
for garment in basic_garments:
    similar_garments = model.most_similar(garment, topn=12)
    garment_list.append(similar_garments)

garment_list = [item[0] for l in garment_list for item in l]
garment_list = set(garment_list)
garment_list

def manual_stopwords(boring_words, model):
    sw = []
    for word in boring_words:
        sw.append(collect_similar(word, model, 10))
    return set([item for l in sw for item in l])

blah_words = ["nice", "pretty", "fantastic", "new", "little", "red", "another", "favorite", "white"]

m_sw = manual_stopwords(blah_words, model)

m_sw

all_sw = set(list(nltk_stopwords) + list(m_sw))



#bigramified_2012 = bigrams_for_all_garments(df2012.tokenized_descs, model, list(all_sw))

#let's save this!
#pd.Series(bigramified_2012).to_pickle('bigrams_2012.pkl')

bigramified_2012 = pd.read_pickle('bigrams_2012.pkl')

#bg_model_2012 = Word2Vec(bigramified_2012)
#bg_model_2012.save('bg_model_2012.pkl')
bg_model_2012 = Word2Vec.load('bg_model_2012.pkl')

trendy_bg = []
for g in basic_garments:
    similar = bg_model_2012.most_similar(g, topn=200)
    bigrams = [item[0] for item in similar if '_' in item[0]]
    trendy_bg.append(bigrams)

trendy_bg 

class Trend(object):


    def __init__(self, phrase, garment_type=None):
        self.phrase = phrase
        self.type = garment_type

    
    def get_tpm_tfm(self, df, start_month, start_year, num_months):
        '''
        Creates a plot showing the term frequency of the Trend
        over a given date range, by month, (using a rolling average over 3 months).
        INPUT: dataframe contaiing dates and descriptions,
                starting month and year, number of months to plot
        OUTPUT: matplotlib object
        '''
        mo = start_month
        yr = start_year
        
        month_year_tuples = []
        for m in range(num_months):
            #create a list of tuples so we can segment our df
            yr_s = str(yr)
            mo_s = str(mo)
            if len(mo_s) == 1:
                mo_s == '0{}'.format(mo_s)

            month_year_tuples.append((yr, mo, 01))

            if mo % 12 == 0:
                yr += 1
                mo = 1
            else:
                mo += 1


        tf_by_mo = []
        tp_by_mo = []
        rnges = []

        #iterate over months
        for tup in month_year_tuples:

            yr_mask = df.year == tup[0]
            df_yr = df[yr_mask]
            mo_mask = df_yr.month == tup[1]
            df_mo = df_yr[mo_mask]
            segment = df_mo.photo_desc
            ct = 0
            total = 0
            for row in segment:
                total += 1
                if self.phrase in row:
                    ct+=1
            tf_by_mo.append(ct)

            
            #so we don't try to divide by 0
            tp_by_mo.append(total + 1)


        tfm = np.array(tf_by_mo)
        tpm = np.array(tp_by_mo)
        my_tuples = month_year_tuples
        return tfm, tpm, my_tuples
    
    def get_tfy_tpy(self, df, start_year, end_year):
        tf_by_yr = []
        tp_by_yr = []
        for yr in xrange(start_year, end_year + 1):
            yr_mask = df.year == yr
            df_yr = df[yr_mask]
            segment = df_yr.photo_desc
            ct = 0
            total = 0
            for row in segment:
                total += 1
                if self.phrase in row:
                    ct+=1
            tf_by_yr.append(ct)

            
            #so we don't try to divide by 0
            tp_by_yr.append(total + 1)


        tfy = np.array(tf_by_yr)
        tpy = np.array(tp_by_yr)
        years = xrange(start_year, end_year + 1)
        return tfy, tpy, years

    
    
    def plot_by_month(self, df, start_month, start_year, num_months, color=None):
        tfm, tpm, month_year_tuples = self.get_tpm_tfm(df, start_month, start_year, num_months)

        y_axis = tfm*1./tpm

        x_axis = coerce_to_datetime(month_year_tuples)
        if color:
            plt.plot(x_axis, y_axis, label=self.phrase, color = color)
        else:
            plt.plot(x_axis, y_axis, label=self.phrase)

    def differences(self, df, start_month, start_year, num_months):
        tfm, tpm, month_year_tuples = self.get_tpm_tfm(df, start_month, start_year, num_months)
        freq_ratio = np.array(tfm * 1.0)/np.array(tpm)
        abs_dif_month_over_month = []
        mag_dif_month_over_month = []
        for i in xrange(0, num_months - 12):
            abs_dif_month_over_month.append(freq_ratio[i + 12] - freq_ratio[i])
        for j in xrange(0, num_months - 12):
            mag_dif_month_over_month.append(abs_dif_month_over_month[j]/freq_ratio[j])
        return freq_ratio, abs_dif_month_over_month, mag_dif_month_over_month, month_year_tuples
    
    def plot_differences(self, df, start_month, start_year, num_months):
        
        freq_ratio, abs_dif_month_over_month, mag_dif_month_over_month, month_year_tuples =                self.differences(df, start_month, start_year, num_months)
        x_axis = coerce_to_datetime(month_year_tuples)
        x_axis = x_axis[12:]
        
        y_axis = mag_dif_month_over_month
        
        plt.plot(x_axis, y_axis, label= 'magnitude changes yoy by month for {}'.format(self.phrase))
        
    def differences_yr(self, df, start_year, end_year):
        tfy, tpy, years = self.get_tfy_tpy(df, start_year, end_year)
        num_yrs = end_year - start_year
        freq_ratio = np.array(tfy * 1.0)/np.array(tpy)
        abs_difs = []
        mag_difs = []
        for i, y in enumerate(xrange(start_year, end_year)):
            abs_dif = freq_ratio[num_yrs] - freq_ratio[i]
            abs_difs.append(abs_dif)
            mag_dif = abs_dif / freq_ratio[i]
            mag_difs.append(("dif {} over {}".format(end_year, y), mag_dif))
            
        return abs_difs, mag_difs
    
               

fleek = Trend("fleek")
fleek.phrase

fleek.plot_by_month(dfZ, 1, 2009, 84, color = 'k')

tiger_print = Trend("tiger print")
tiger = Trend("tiger")

tiger_print.plot_by_month(dfZ, 1, 2009, 84, color = 'k')
tiger.plot_by_month(dfZ, 1, 2009, 84, color = 'grey')

yellow = Trend("yellow")
yellow.plot_by_month(dfZ, 1, 2009, 94, color = 'gold')

def show_plot(highlight_year=None):
    plt.legend()
    plt.xticks(rotation = -35, ha='left')
    if highlight_year:
        plt.axvspan(pd.datetime(highlight_year, 1, 1), pd.datetime(highlight_year + 1, 1, 1), color='grey', alpha=0.6)
        plt.axvspan(pd.datetime(highlight_year, 1, 1), pd.datetime(highlight_year + 2, 1, 1), color='grey', alpha=0.3)
    seaborn.set(rc={'figure.facecolor':'white'})
    plt.get_cmap('spring')
    plt.ylabel("term frequency / total posts")
    plt.show()
    
    
    

show_plot(2016)

trends_2012 = []
for l in trendy_bg:
    for item in l:
        item_split = item.split('_')
        item = Trend('{} {}'.format(item_split[0], item_split[1]))
        trends_2012.append(item)

top_trends = set([t.phrase for t in trends_2012])

top_trends

descriptors = []
for item in top_trends:
    item_s = item.split()
    descriptors.append(item_s[0])
count_descriptors = Counter(descriptors)

count_descriptors


        


long_dress = Trend("long dress")




long_dress.differences_yr(dfZ, 2009, 2013)

interesting_bigrams = ['maxi_dress', 'maxi_skirt', 'maxi', 'peplum_top', 'peplum_dress', 'peplum_skirt',  'peplum', 'crop_top','cropped_top', 'cropped_sweater', 'cropped_blazer'] 

#note to self: make Trendification a function
interesting = []
for item in interesting_bigrams:

    if '_' in item:
        item_split = item.split('_')
        item = Trend('{} {}'.format(item_split[0], item_split[1]))
    else:
        item = Trend('{}'.format(item))
    interesting.append(item)
    

[t.phrase for t in interesting]

for t in interesting[:3]:
    t.plot_by_month(dfZ, 1, 2009, 60)
show_plot(2012) 

for t in interesting[3:7]:
    
    t.plot_by_month(dfZ, 1, 2009, 60)
show_plot(2012) 


interesting[7].plot_by_month(dfZ, 1, 2009, 60)

show_plot(2012) 







def coerce_to_datetime(series):
    series_2 = []
    for item in series:
        try:
            s = "{}, {}, {}".format(item[0], item[1], item[2])
            series_2.append(s)
        except TypeError:
            series_2.append('2008, 03, 01')
    series_2 = pd.Series(series_2)
    series_3 = pd.to_datetime(series_2)
    return series_3



def get_tf(df, phrase, start_month, start_year, num_months):
    '''
    INPUT: dataframe containing dates and descriptions,
            starting month and year, number of months to search
    OUTPUT: vector as list
    '''
    yr = start_year
    mo = start_month
    tf_by_mo = []
    tp_by_mo = []


    for m in range(num_months):
        print yr
        print mo
#         mask_year = (df.year == yr)
#         df = df[mask_year]
#         mask_month = (df.month == mo)
#         df = df[mask_month]
        if mo % 12 == 0:
            yr += 1
            mo = 1
        else:
            mo += 1
#         df = df.photo_desc
#         total = 0
#         ct = 0
        for row in df.iterrows():
            print row
#     print phrase
#     print "Term freq by month: ", tf_by_mo 
#     print "Total term mentions over this period: ", sum(tf_by_mo)
#     print "proportion of posts using this term, by month: ", np.array(tf_by_mo)/np.array(tp_by_mo)







for line in trendy_bg:
    for item in line:
        item_split = item.split('_')
        get_tf(dfZ, '{} {}'.format(item[0], item[1]), 6, 2008, 24)

class Trend(object):

    def __init__(self, phrase, garment_type=None):
        self.phrase = phrase
        self.type = garment_type


    def plot_over_time(self, df, date_begin, num_weeks):
        '''
        Creates a plot showing the term frequency of the Trend
        over a given date range
        INPUT: dataframe contaiing dates and descriptions,
                start and end date in std format ['1980-01-15']
        OUTPUT: matplotlib object
        '''
        start = pd.to_datetime(date_begin)
        term_frequencies_by_day = []
        total_posts_by_day = []
        tf_by_wk = []
        tp_by_wk = []
        rnges = []

        for week in range(num_weeks):
            rnges.append(pd.date_range(start, start + pd.Timedelta(days=7)))
            start = start + pd.Timedelta(days=7)

        for rng in rnges:


            for a_date in rng:
                mask = df.datetime == a_date
                segment = df[mask]
                segment = segment.photo_desc
                ct = 0
                total = 0
                for row in segment:
                    total += 1
                    if self.phrase in row:
                        ct+=1
                term_frequencies_by_day.append(ct)
                total_posts_by_day.append(total)

            wk_tf = sum(term_frequencies_by_day)
            wk_total = sum(total_posts_by_day)
            tf_by_wk.append(wk_tf)
            tp_by_wk.append(wk_total)

            term_frequencies_by_day = []
            total_posts_by_day = []




        tfw = np.array(tf_by_wk)
        tpw = np.array(tp_by_wk)
        dates_by_week = [rng[0] for rng in rnges]

        data = tfw*1./tpw
        #density = gaussian_kde(data)
        xs = dates_by_week
        #density.covariance_factor = lambda : .25
        #density._compute_covariance()
        #plt.plot(xs,density(xs))


        #plot dates on x axis, frequency on y-axis
        #moving_ave = np.convolve(tfw*1./tpw, np.ones(4)/4)[1:num_weeks+1]
        plt.plot(dates_by_week, tfw*1./tpw, label=self.phrase)






    def plot_by_week(self, df, date_begin, num_weeks):
        '''
        Creates a plot showing the term frequency of the Trend
        over a given date range, by week, using a rolling average over 4 weeks.
        INPUT: dataframe contaiing dates and descriptions,
                start and end date in std format ['1980-01-15']
        OUTPUT: matplotlib object
        '''
        start = pd.to_datetime(date_begin)
        term_frequencies_by_day = []
        total_posts_by_day = []
        tf_by_wk = []
        tp_by_wk = []
        rnges = []

        for week in range(num_weeks):
            rnges.append(pd.date_range(start, start + pd.Timedelta(days=7)))
            start = start + pd.Timedelta(days=7)

        for rng in rnges:


            for a_date in rng:
                mask = df.datetime == a_date
                segment = df[mask]
                segment = segment.photo_desc
                ct = 0
                total = 0
                for row in segment:
                    total += 1
                    if self.phrase in row:
                        ct+=1
                term_frequencies_by_day.append(ct)
                total_posts_by_day.append(total)

            wk_tf = sum(term_frequencies_by_day)
            wk_total = sum(total_posts_by_day)
            tf_by_wk.append(wk_tf)
            tp_by_wk.append(wk_total)

            term_frequencies_by_day = []
            total_posts_by_day = []




        tfw = np.array(tf_by_wk)
        tpw = np.array(tp_by_wk)
        dates_by_week = [rng[0] for rng in rnges]

        data = tfw*1./tpw
        density = gaussian_kde(data)
        xs = dates_by_week
        density.covariance_factor = lambda : .25
        density._compute_covariance()
        plt.plot(xs,density(xs))


        #plot dates on x axis, frequency on y-axis
        #moving_ave = np.convolve(tfw*1./tpw, np.ones(4)/4)[1:num_weeks+1]
        plt.plot(dates_by_week, tfw*1./tpw, label=self.phrase)


    def plot_by_month(self, df, start_month, start_year, num_months):
        '''
        Creates a plot showing the term frequency of the Trend
        over a given date range, by month, (using a rolling average over 3 months).
        INPUT: dataframe contaiing dates and descriptions,
                starting month and year, number of months to plot
        OUTPUT: matplotlib object
        '''
        start_year == str(start_year)
        if len(str(start_month)) == 1:
            start_month = str(0) + str(start_month)
        else:
            start_month = str(start_month)

        start = pd.to_datetime('{}{}01'.format(start_year, start_month))

        term_frequencies_by_day = []
        total_posts_by_day = []
        tf_by_mo = []
        tp_by_mo = []
        rnges = []

        for month in range(num_months):
            #create a list of date ranges, one item for each months
            rnges.append(pd.date_range(start, start + pd.Timedelta(days=30)))
            start = start + pd.Timedelta(days=30)

        #iterate over months
        for rng in rnges:
            #iterate over dates in month
            for a_date in rng:
                mask = df.datetime == a_date
                segment = df[mask]
                segment = segment.photo_desc
                ct = 0
                total = 0
                for row in segment:
                    total += 1
                    if self.phrase in row:
                        ct+=1
                term_frequencies_by_day.append(ct)
                total_posts_by_day.append(total)

            mo_tf = sum(term_frequencies_by_day)
            mo_total = sum(total_posts_by_day) + 1
            tf_by_mo.append(mo_tf)
            tp_by_mo.append(mo_total)
            #reset term frequencies by day
            term_frequencies_by_day = []
            total_posts_by_day = []


        tfm = np.array(tf_by_mo)
        tpm = np.array(tp_by_mo)


        y_axis = tfm*1./tpm
        #density = gaussian_kde(y_axis)
        x_axis = [rng[0] for rng in rnges]
        #density.covariance_factor = lambda : .25
        #density._compute_covariance()
        #plt.plot(xs,density(xs))

        #rolling average. first term mean of months 1 & 2; last (nth) term mean of n-1 & n
        #all others mean of k-1, k, k+1
        #y_axis[0] = np.mean(y_axis[0:1])
        #y_axis[-1] = np.mean(y_axis[-2:-1])
        #y_axis[1:-2] = [np.mean(y_axis[i-1:i+1]) for i in range(len(y_axis[1:-2]))]

        #plot dates on x axis, frequency on y-axis
        #moving_ave = np.convolve(tfw*1./tpw, np.ones(4)/4)[1:num_weeks+1]
        print tfm

        plt.plot(x_axis, y_axis, label=self.phrase)
        return tfm
    

dfZ.year







tokens = df2009.tokenized_descs

tokens

def bigrams_colloc(tokens):
    finder = nltk.BigramCollocationFinder.from_words(tokens)
    bigram_measure = nltk.collocations.BigramAssocMeasures()
    return finder, bigram_measure

def bigrams_standard(tokens):
    bigrams = nltk.bigrams(tokens)
    fdist = nltk.FreqDist(bigrams)
    return bigrams, fdist
    

dress = Trend("dress")
pants = Trend("pants")
shoes = Trend("shoes")

def six_mo_plot(df, year, start_q, term):
    q1 = quarter_dfs['{}_q{}'.format(year, start_q)]
    if start_q == 4:
        q2 = quarter_dfs['{}_q{}'.format(year + 1, 1)]
    else:
        q2 = quarter_dfs['{}_q{}'.format(year, start_q + 1)]

    df_this = pd.concat([q1, q2])
    
    tokens = tokenize(df_this)
    finder, bigram_measure = bigrams_colloc(tokens)
    
    top_freq = []
    while len(top_freq) < 10:
        for phrase in finder.nbest(bigram_measure.raw_freq, 100000):
            if str(phrase[1]) == term and str(phrase[0]) not in nltk_sw:
                top_freq.append(phrase)      
        
    for ind in range(10):
        this_trend = top_freq[ind]
        this_trend = "{} {}".format(this_trend[0], this_trend[1])
        this_trend_ob = Trend(this_trend)

        print this_trend_ob.phrase

        this_trend_ob.plot_over_time(df, '2008-06-30', 156)
        plt.legend()
        plt.xticks(rotation=35)
        plt.axvspan(df_this.datetime.min(), df_this.datetime.max(), color='red', alpha=0.3)
        plt.show()

        
    

dress.plot_by_month(dfY, 7, 2008, 60)
plt.show()

maxi = Trend("maxi")
maxi.plot_by_month(dfY, 7, 2008, 60)
maxi_dress = Trend("maxi dress")
maxi_dress.plot_by_month(dfY, 7, 2008, 60)
plt.show()









blah_words = ["nice", "pretty", "fantastic", "new", "little", "red", "another", "favorite"]

model.most_similar("dress")

def manual_stopwords(boring_words, model):
    sw = []
    for word in boring_words:
        sw.append(collect_similar(word, model, 10))
    return set([item for l in sw for item in l])

man_sw = manual_stopwords(blah_words, model)

man_sw

all_sw = list(nltk_stopwords) + list(man_sw)

all_tokens = dfZ.tokenized_descs

#bigrams = bigrams_for_all_garments(all_tokens, model, all_sw)

bigrams2009 = bigrams_for_all_garments(df2009.tokenized_descs, model, all_sw)

model_bg_2009 = Word2Vec(bigrams2009)



def get_bigrams(model, term):
    bigrams = []
    bigrams_split = []
    
    for item in model.most_similar(term, topn=10000):
        if '_'.format(term) in item[0]:
            item_split = item[0].split('_')
            bigrams.append(item[0])
            bigrams_split.append('{} {}'.format(item_split[0], item_split[1]))

    return bigrams, bigrams_split

dress_bg_2009 = get_bigrams(model_bg_2009, "dress")

top_100 = dress_bg_2009[0][:100]


for bigram in top_100:
    print bigram
    item_split = bigram.split('_')
    bigram = Trend('{} {}'.format(item_split[0], item_split[1]))
    tfm = bigram.plot_by_month(dfY, 6, 2008, 48)
    if sum(tfm) > 500:
        plt.show()
    else:
        plt.close()

maxi_dress = Trend("maxi dress")
maxi_dress.plot_by_month(dfZ, 1, 2009, 60)
maxi_skirt = Trend("maxi skirt")
maxi_skirt.plot_by_month(dfZ, 1, 2009, 60)
maxi = Trend("maxi")
maxi.plot_by_month(dfZ, 1, 2009, 60)
    
show_plot(2012)



