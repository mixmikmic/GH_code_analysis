get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext rpy2.ipython')

import requests
import pandas as pd
import numpy as np
from datetime import datetime

from api_functions import getProblemDataFromContest
from api_functions import getContestList

df_problem_ratings = pd.DataFrame.from_csv('problem_ratings.csv', index_col=None)
df_problem_info = pd.DataFrame.from_csv('problem_data.csv', index_col=None)

df_problems = pd.merge(df_problem_ratings, df_problem_info)
df_rhist = pd.DataFrame.from_csv('rating_histories.csv', index_col=None)
df_rhist = df_rhist.rename(columns={'contestId':'contestID'})

# parsing problem tags
from re import sub
from re import compile
dict_tag = []

regexp = compile('\(.+?\)')
regexp2 = compile('\[.+?\]')
regexp3 = compile('\(.+?\)')
regexp4 = compile('\".+?\"')
with open('problem_data.csv') as f:
    lines = f.readlines()
    headers = lines[0].strip().split(',')
    tag_idx = headers.index('tags')
    
    for line in lines[1:]:
        oldline = line
        sline = regexp.sub('', line.strip())
        sline = regexp2.sub('', sline)
        sline = regexp3.sub('', sline)
        sline = regexp4.sub('', sline)
        sline = sline.split(',')
        contestID = sline[0]

        division = sline[2]
        problemID = sline[5]

        if ',"[' in line:
            tags = line.strip().split(',"[')[1]
        elif ',[' in line:
            tags = line.strip().split(',[')[1]
        tags = tags.split(']')[0]
        tags = tags.split(', ')

        for tag in tags:
            dict_tag.append(
                {
                    'contestID': int(contestID),
                    'problemID': problemID,
                    'division': int(division),
                    'tag': tag
                }
                )
df_tags = pd.DataFrame.from_dict(dict_tag)
df_tags = pd.merge(df_tags, df_problem_ratings)

get_ipython().run_cell_magic('R', '-i df_problems -i df_rhist -i df_tags', '# [1] "contestID"        "problemID"        "problemRating"    "contestName"     \n# [5] "division"         "name"             "points"           "startTimeSeconds"\n# [9] "tags"             "type"   \nlibrary(ggplot2)\nlibrary(plotly)\n\ndf <- df_problems\n\ndf$division <- factor(df$division, levels=c(1,2,12))\ndf$bin <- cut(df$points, c(0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000))\ndf$type <- \'other\'\ndf$adjusted_elo <- df$problemRating\nbgalpha <- .2\n\n# annotate max and min problem ELOs\ntapply(1:nrow(df), df$contestID, function(idx){\n    ismax <- df[idx, \'problemRating\'] == max(df[idx, \'problemRating\'])\n    df$type[idx[ismax]] <<- \'hardest problem in contest\'\n    \n    ismin <- df[idx, \'problemRating\'] == min(df[idx, \'problemRating\'])\n    df$type[idx[ismin]] <<- \'easiet problem in contest\'\n    \n    df$adjusted_elo[idx] <<- df$adjusted_elo[idx] - min(df$adjusted_elo[idx])\n})\n\n# create dict of contest to division\ndict_contestID_division <- unique(df[, c(\'contestID\' ,\'division\')])\nrownames(dict_contestID_division) <- as.character(dict_contestID_division$contestID)\n\n# average user rating per contest\naverageRating <- tapply(1:nrow(df_rhist), df_rhist$contestID, function(idx){\n    contestID <- df_rhist[idx[1], \'contestID\']\n    averageRating <- median(df_rhist[idx, \'newRating\'])\n    division <- dict_contestID_division[as.character(contestID), \'division\']\n    \n    data.frame(contestID=contestID,\n              averageRating=averageRating,\n              division=division)\n})\ndf_averageRating <- do.call(rbind, averageRating)\n\n# filter out combined contests\ndf_averageRating <- df_averageRating[df_averageRating$division != 12,]\ndf$problemID_simple <- substr(df$problemID, 1, 1)\ndf$divsion <- factor(df$division)\n\n# Figure: point value vs. problem rating, violin plot\ndf2 <- df[!is.na(df$points),]\ndf2$division <- gsub(\'12\', \'1 + 2 Combined\', df2$division)\ndf2$division <- gsub(\'1\', \'Div. 1\', df2$division)\ndf2$division <- gsub(\'2\', \'Div. 2\', df2$division)\ndf2$division <- factor(df2$division, levels=c(\'Div. 2\', \'Div. 1\', \'Div. 1 + Div. 2 Combined\'))\nfig_points_vs_rating <- ggplot(df2) +\n    geom_violin(aes(x=bin, y=problemRating)) + \n    annotate("rect", ymin=1200, ymax=1399, xmin=-Inf, xmax=Inf, color=NA, fill=\'green\', alpha=bgalpha) +\n    annotate("rect", ymin=1400, ymax=1599, xmin=-Inf, xmax=Inf, color=NA, fill=\'#30DBCA\', alpha=bgalpha) +\n    annotate("rect", ymin=1600, ymax=1899, xmin=-Inf, xmax=Inf, color=NA, fill=\'#3094DB\', alpha=bgalpha) +\n    annotate("rect", ymin=1900, ymax=2199, xmin=-Inf, xmax=Inf, color=NA, fill=\'#B930DB\', alpha=bgalpha) +\n    annotate("rect", ymin=2200, ymax=2299, xmin=-Inf, xmax=Inf, color=NA, fill=\'#FFEA4D\', alpha=bgalpha) +\n    annotate("rect", ymin=2300, ymax=2399, xmin=-Inf, xmax=Inf, color=NA, fill=\'#FFBF00\', alpha=bgalpha) +\n    annotate("rect", ymin=2400, ymax=2599, xmin=-Inf, xmax=Inf, color=NA, fill=\'#FF7E61\', alpha=bgalpha) +\n    annotate("rect", ymin=2600, ymax=2899, xmin=-Inf, xmax=Inf, color=NA, fill=\'#FF4117\', alpha=bgalpha) +\n    annotate("rect", ymin=2900, ymax=Inf, xmin=-Inf, xmax=Inf, color=NA, fill=\'#CC0000\', alpha=bgalpha) +\n    geom_violin(aes(x=bin, y=problemRating)) + \n    geom_jitter(aes(x=bin, y=problemRating), width = .6, size=.5, alpha=.1, color=\'blue\') +\n    facet_wrap(~division) +\n\ttheme(axis.text.x = element_text(angle = 45, hjust=1, vjust=1)) +\n    labs(x="Points Assigned to Problem", y="Problem Rating")\n\n# Figure: histogram of problem ratings for various divisions\nc <- ggplot(df)\nfig_points_histogram <- c + \n    annotate("rect", xmin=1200, xmax=1399, ymin=-Inf, ymax=Inf, color=NA, fill=\'green\', alpha=bgalpha) +\n    annotate("rect", xmin=1400, xmax=1599, ymin=-Inf, ymax=Inf, color=NA, fill=\'#30DBCA\', alpha=bgalpha) +\n    annotate("rect", xmin=1600, xmax=1899, ymin=-Inf, ymax=Inf, color=NA, fill=\'#3094DB\', alpha=bgalpha) +\n    annotate("rect", xmin=1900, xmax=2199, ymin=-Inf, ymax=Inf, color=NA, fill=\'#B930DB\', alpha=bgalpha) +\n    annotate("rect", xmin=2200, xmax=2299, ymin=-Inf, ymax=Inf, color=NA, fill=\'#FFEA4D\', alpha=bgalpha) +\n    annotate("rect", xmin=2300, xmax=2399, ymin=-Inf, ymax=Inf, color=NA, fill=\'#FFBF00\', alpha=bgalpha) +\n    annotate("rect", xmin=2400, xmax=2599, ymin=-Inf, ymax=Inf, color=NA, fill=\'#FF7E61\', alpha=bgalpha) +\n    annotate("rect", xmin=2600, xmax=2899, ymin=-Inf, ymax=Inf, color=NA, fill=\'#FF4117\', alpha=bgalpha) +\n    annotate("rect", xmin=2900, xmax=Inf,  ymin=-Inf, ymax=Inf, color=NA, fill=\'#CC0000\', alpha=bgalpha) +\n    geom_freqpoly(aes(x=problemRating, ..density.., linetype=division, group=division, color=division), binwidth=100) +\n    scale_color_manual(values = c(\'black\', \'black\', \'gray\')) +\n\ttheme(axis.text.x = element_text(angle = 45, hjust=1, vjust=1))\n\n# Figure: problem and user ratings over time\ncolor_scale <- c(\'hardest problem in contest\' = \'red\', \n                 \'easiet problem in contest\'=\'blue\',\n                 \'other\'=\'gray\')\ndf$division <- as.character(df$division)\ndf <- df[df$division != \'12\',]\ndf <- df[df$problemRating != 5000,]\n\nfig_contestID_v_rating <-  \n    ggplot(df) + \n    geom_point(aes(x=contestID, y=problemRating, color=type), alpha=.2, size=2)  +\n    geom_line(data=df_averageRating, aes(x=contestID, y=averageRating, group=division, linetype=division), \n              color=\'black\', size=1) + \n    scale_color_manual(values = color_scale) +\n    facet_wrap(~division, drop=TRUE, ncol=1) +\n    scale_alpha(range=c(0,1)) + \n    theme(legend.position = \'bottom\')\n\n\n# Figure: problem rating vs. tags, violin plot\n# first reorder the dataframe by median difficulty\nsorting <- tapply(1:nrow(df_tags), df_tags$tag, function(idx){\n    data.frame(tag=df_tags[idx, \'tag\'][1], rating = median(df_tags[idx, \'problemRating\']))\n})\nsorting <- do.call(rbind,sorting)\nsorting <- sorting[order(sorting$rating),]\ndf_tags$tag <- factor(df_tags$tag, levels=sorting$tag)\n\nfig_tags <- ggplot() +\n    geom_violin(data=df_tags, aes(x=tag, y=problemRating), alpha=1) + \n    annotate("rect", ymin=1200, ymax=1399, xmin=-Inf, xmax=Inf, color=NA, fill=\'green\', alpha=bgalpha) +\n    annotate("rect", ymin=1400, ymax=1599, xmin=-Inf, xmax=Inf, color=NA, fill=\'#30DBCA\', alpha=bgalpha) +\n    annotate("rect", ymin=1600, ymax=1899, xmin=-Inf, xmax=Inf, color=NA, fill=\'#3094DB\', alpha=bgalpha) +\n    annotate("rect", ymin=1900, ymax=2199, xmin=-Inf, xmax=Inf, color=NA, fill=\'#B930DB\', alpha=bgalpha) +\n    annotate("rect", ymin=2200, ymax=2299, xmin=-Inf, xmax=Inf, color=NA, fill=\'#FFEA4D\', alpha=bgalpha) +\n    annotate("rect", ymin=2300, ymax=2399, xmin=-Inf, xmax=Inf, color=NA, fill=\'#FFBF00\', alpha=bgalpha) +\n    annotate("rect", ymin=2400, ymax=2599, xmin=-Inf, xmax=Inf, color=NA, fill=\'#FF7E61\', alpha=bgalpha) +\n    annotate("rect", ymin=2600, ymax=2899, xmin=-Inf, xmax=Inf, color=NA, fill=\'#FF4117\', alpha=bgalpha) +\n    annotate("rect", ymin=2900, ymax=Inf, xmin=-Inf, xmax=Inf, color=NA, fill=\'#CC0000\', alpha=bgalpha) +\n    geom_violin(data=df_tags, aes(x=tag, y=problemRating), alpha=1) + \n    geom_jitter(data=df_tags, aes(x=tag, y=problemRating), width = .6, size=.5, alpha=.2, color=\'blue\') +\n\ttheme(axis.text.x = element_text(angle = 45, hjust=1, vjust=1))\n\n# Print plots to PDF files\npdf(\'fig_contestID_v_rating.pdf\', width=8, height=12); print(fig_contestID_v_rating); dev.off()\npdf(\'fig_points_vs_rating.pdf\', width=10, height=5); print(fig_points_vs_rating); dev.off()\npdf(\'fig_points_histogram.pdf\', width=10, height=5); print(fig_points_histogram); dev.off()\npdf(\'fig_tags.pdf\', width=15, height=5); print(fig_tags); dev.off()\n')

# read in probable duplicate questions
duplicates = []
with open('problem_duplicates.csv') as f:
    lines = f.readlines()
    for line in lines[1:]:
        line = line.strip().split(',')
        if line[0] != line[2] and line[2] != line[3]:
            duplicates.append(line)

df_dup = []

cnt = 0
for dup in duplicates:
    p1 = df_problems.loc[(df_problems.contestID == int(dup[0])) & (df_problems.problemID == dup[1])]
    p2 = df_problems.loc[(df_problems.contestID == int(dup[2])) & (df_problems.problemID == dup[3])]
    if p1.shape[0] > 0 and p2.shape[0] > 0:
        if p1.loc[p1.index[0], 'division'] == 2:
            p1, p2 = p2, p1
        
        if abs(p1.loc[p1.index[0], 'contestID'] - p2.loc[p2.index[0], 'contestID']) > 1:
            continue
        
        if p1.loc[p1.index[0], 'problemID'] > p2.loc[p2.index[0], 'problemID']:
            continue
            
        data = {
            'd1_contestID':p1.loc[p1.index[0], 'contestID'],
            'd2_contestID':p2.loc[p2.index[0], 'contestID'],
            'd1_problemID':p1.loc[p1.index[0], 'problemID'],
            'd2_problemID':p2.loc[p2.index[0], 'problemID'],
            'd1_elo':p1.loc[p1.index[0], 'problemRating'],
            'd2_elo':p2.loc[p2.index[0], 'problemRating']
        }
        
        df_dup.append(data)
        
        cnt += 1
        if cnt == 100:
            break

df_dup = pd.DataFrame.from_dict(df_dup)

get_ipython().run_cell_magic('R', '-i df_dup', "df <- df_dup\n\nratings <- c(1200, 1400, 1600, 1900, 2200, 2300, 2400, 2600, 2900, Inf)\ncolorscales = c(\n    '0' = 'gray',\n    '1' = 'green',\n    '2' = '#30DBCA',\n    '3' = '#3094DB',\n    '4' = '#B930DB',\n    '5' = '#FFEA4D',\n    '6' = '#FFBF00',\n    '7' = '#FF7E61',\n    '8' = '#FF4117',\n    '9' = '#CC0000'\n)\nnames(colorscales) <- unique(df$color)\n\ndf$color <- cut(df$d1_elo, ratings)\nc <- ggplot(df, aes(x=d1_elo, y=d2_elo))  +\n    geom_point(alpha=.5, aes(color=color), size=3) +\n    geom_abline(intercept=0, slope=1, size=.2, color='black') +\n    scale_color_manual(values = colorscales) +\n    theme(legend.position = 'None') +\n    labs(x='Div. 1 Problem ELO Score', y='Div. 2 Problem ELO Score')\n#ggplotly()\npdf('fig_d1_v_d2.pdf', width=5, height=5)\nprint(c)\ndev.off()")

