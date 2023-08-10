get_ipython().magic('matplotlib inline')
import moviepy.editor
import numpy as np
import cv2
import matplotlib.pyplot as plt

def detect_motion(clip,last_fr=None):
    fr_motion=list()
    for f in clip.iter_frames(fps=10,dtype='uint32', progress_bar=1):
        if last_fr is not None:
            fr_diff = f-last_fr
            fr_motion += [fr_diff.sum()]
            del(fr_diff)
            del last_fr
        last_fr=f.copy()
    return fr_motion
    
clip=moviepy.editor.VideoFileClip('sample.mp4',target_resolution=(240,None))
fr_list=detect_motion(clip)


#len(fr_list)
#max_fr=max(fr_list) 
#scaling the values between o to 100
#fr_list_scaled= [i*100/max_fr for i in fr_list]

#fr_num= np.arange(.1, clip.duration , .1 )
#len(fr_num)

plt.plot(fr_list )
plt.show()

clip.duration

kaiser_func = np.kaiser(int(clip.duration),10)
plt.plot(kaiser_func)

motion_act_unscaled = np.convolve(kaiser_func/kaiser_func.sum(),fr_list,mode='same')

len(motion_act_unscaled)

max_val=max(motion_act_unscaled) 

#scaling the values between o to 100
motion_act=motion_act_unscaled*100/max_val

print(motion_act)

plt.plot(motion_act)

def get_filtered_peaks(smooth_motion_activity):
    #find peaks from the curve
    increases = np.diff(smooth_motion_activity)[:-1]>=0
    decreases = np.diff(smooth_motion_activity)[1:]<=0
    peaks_position = (increases * decreases).nonzero()[0]
    print(peaks_position)
    peaks_value = smooth_motion_activity[peaks_position]
    peaks_position = peaks_position[peaks_value>np.percentile(peaks_value,10)]
    
    #filter two close (100 frames apart) peaks
    final_peaks_position=[peaks_position[0]]
    for fr_num in peaks_position:
        if (fr_num - final_peaks_position[-1]) < 100:
            if smooth_motion_activity[fr_num] > smooth_motion_activity[final_peaks_position[-1]]:
                final_peaks_position[-1] = fr_num
        else:
            final_peaks_position.append(fr_num)

    final_times = [i/10 for i in final_peaks_position]
    return final_times

motion_peaks_times=get_filtered_peaks(motion_act)
print 'time of peaks (in secs) =', motion_peaks_times

final = moviepy.editor.concatenate([clip.subclip(max(t-5,0),min(t+5, clip.duration))
                     for t in motion_peaks_times])
#final.ipython_display()
final.to_videofile('summary_by_motion_detection.mp4')





