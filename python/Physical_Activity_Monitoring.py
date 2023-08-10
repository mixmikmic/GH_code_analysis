get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
custom_style = {
            'grid.color': '0.8',
            'grid.linestyle': '--',
            'grid.linewidth': 0.5,
}
sns.set_style(custom_style)

columns = ['timestamp', 'activity_id', 'heart_rate',
           'IMU_HAND_TEMP_C',
           'IMU_HAND_16G_2',
           'IMU_HAND_16G_3',
           'IMU_HAND_16G_4',
           'IMU_HAND_6G_5',
           'IMU_HAND_6G_6',
           'IMU_HAND_6G_7',
           'IMU_HAND_GYRO_8',
           'IMU_HAND_GYRO_9',
           'IMU_HAND_GYRO_10',
           'IMU_HAND_MAG_11',
           'IMU_HAND_MAG_12',
           'IMU_HAND_MAG_13',
           'IMU_HAND_ORIENT_14',
           'IMU_HAND_ORIENT_15',
           'IMU_HAND_ORIENT_16',
           'IMU_HAND_ORIENT_17',
           'IMU_CHEST_TEMP_C',
           'IMU_CHEST_16G_2',
           'IMU_CHEST_16G_3',
           'IMU_CHEST_16G_4',
           'IMU_CHEST_6G_5',
           'IMU_CHEST_6G_6',
           'IMU_CHEST_6G_7',
           'IMU_CHEST_GYRO_8',
           'IMU_CHEST_GYRO_9',
           'IMU_CHEST_GYRO_10',
           'IMU_CHEST_MAG_11',
           'IMU_CHEST_MAG_12',
           'IMU_CHEST_MAG_13',
           'IMU_CHEST_ORIENT_14',
           'IMU_CHEST_ORIENT_15',
           'IMU_CHEST_ORIENT_16',
           'IMU_CHEST_ORIENT_17',
           'IMU_ANKLE_TEMP_C',
           'IMU_ANKLE_16G_2',
           'IMU_ANKLE_16G_3',
           'IMU_ANKLE_16G_4',
           'IMU_ANKLE_6G_5',
           'IMU_ANKLE_6G_6',
           'IMU_ANKLE_6G_7',
           'IMU_ANKLE_GYRO_8',
           'IMU_ANKLE_GYRO_9',
           'IMU_ANKLE_GYRO_10',
           'IMU_ANKLE_MAG_11',
           'IMU_ANKLE_MAG_12',
           'IMU_ANKLE_MAG_13',
           'IMU_ANKLE_ORIENT_14',
           'IMU_ANKLE_ORIENT_15',
           'IMU_ANKLE_ORIENT_16',
           'IMU_ANKLE_ORIENT_17']

df = pd.read_csv('/home/pybokeh/Downloads/PAMAP2_Dataset/Protocol/subject101.dat', sep=' ', 
                 header=None, names=columns).query("activity_id != 0")

df.head()

id_to_activity = {0: 'other',
                  1: 'lying',
                  2: 'sitting',
                  3: 'standing',
                  4: 'walking',
                  5: 'running',
                  6: 'cycling',
                  7: 'Nordic walking',
                  9: 'watching TV',
                  10: 'computer work',
                  11: 'car driving',
                  12: 'ascending stairs',
                  13: 'descending stairs',
                  16: 'vacuum cleaning',
                  17: 'ironing',
                  18: 'folding laundry',
                  19: 'house cleaning',
                  20: 'playing soccer',
                  24: 'rope jumping'}

df['activity'] = df.activity_id.map(id_to_activity)

df.head()

df.isnull().any()

df.shape

df.dropna(inplace=True)
df.shape

df.boxplot(column='heart_rate', by='activity')
plt.xticks(rotation=-90)
sns.despine()
plt.show()

df[['timestamp', 'heart_rate', 'activity']].query("activity == 'running'").describe()

df.query("activity == 'descending stairs'").plot.line(x="timestamp", y="heart_rate")

