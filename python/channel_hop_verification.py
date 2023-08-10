from datetime import datetime
import matplotlib.pyplot as plt
from ipywidgets import widget
import re

def load_time_series(file_name):
    d_list = []
    time_offset = None
    with open(file_name) as f:
        for l in f:
            if not len(l):
                # Blank line, ignore
                continue
            if 'hopper' not in l:
                # Sniffed frame output, ignore
                continue
            d = {}
            if not time_offset:
                time_offset = float(l[:17])
            try:
                d["timestamp"] = float(l[:17])-time_offset
            except ValueError:
                # Sometimes this happen as the outputs from the sniffer is corrupted
                continue
            d["message"] = l[18:]
            d_list += [d]
    return d_list

def filter_time_series_hopper(d_list):
    out_list = []
    for l in (d for d in d_list if "next time" in d["message"]):
        try:
            l["channel"] = int(l["message"][21:23])
            l["next_dur"] = int(l["message"].split('gets')[1][1:-4])
        except:
            # Corruption again
            continue
        out_list += [l]
    return out_list

def calculate_hopper_duration_percent(d_list):
    channel_hopping_times = [None]*13
    for d in d_list:
        channel_hopping_times[d["channel"]-1]=d["next_dur"]
        if None not in channel_hopping_times:
            # Calculate percentage
            d["percentage"] = d["next_dur"] / sum(channel_hopping_times)*100
        else:
            d["percentage"] = 0
    return d_list

def plot_time_series_hopper(d_list,loc):
    plt.figure(figsize=(16,16))
    inchart = plt.subplot('211')
    channel_averages = []
    for i in range(1,14):
        plt.plot([d["timestamp"] for d in d_list if d["channel"] == i],[d["next_dur"] for d in d_list if d["channel"] == i],label="Channel %d"%i)
        channel_averages += [sum([d["next_dur"] for d in d_list if d["channel"] == i])/len([d["next_dur"] for d in d_list if d["channel"] == i])]
    inchart.legend()
    inchart.set_xlabel("Time (s)")
    inchart.set_ylabel("Sniff duration (ms)")
    inchart.set_title("Sniff duration of each channel - "+loc)
    avchart = plt.subplot('212')
    for i in range(1,14):
        avchart.plot([d["timestamp"] for d in d_list if d["channel"] == i],[d["percentage"] for d in d_list if d["channel"] == i],label="Channel %d"%i)
    avchart.legend()
    avchart.set_xlabel("Time (s)")
    avchart.set_ylabel("Sniff duration (%)")
    avchart.set_title("Sniff percentage of each channel - "+loc)
    plt.show()

hopper_msg_list = load_time_series("smart_hopping_chiwah_insideroom.log")
hopper_msg_list = filter_time_series_hopper(hopper_msg_list)
hopper_msg_list = calculate_hopper_duration_percent(hopper_msg_list)
plot_time_series_hopper(hopper_msg_list,"Inside Chi Wah study room")

hopper_msg_list = load_time_series("smart_hopping_chiwah_outsideroom.log")
hopper_msg_list = filter_time_series_hopper(hopper_msg_list)
hopper_msg_list = calculate_hopper_duration_percent(hopper_msg_list)
plot_time_series_hopper(hopper_msg_list,"Outside Chi Wah study room")

import re
sniffed_frames = []
with open("smart_hopping_chiwah_outsideroom.log") as f:
    for l in f:
        if re.match('^.*?\d{10} CH\d{2} RI-\d+ \d{2} \d{2}',l):
            frame_line = l[21:]
            sniffed_frames += [frame_line]

# Filter invalid types
validtypes = ['00', # Association request
              '01', # Association response
              '02', # Reassociation request
              '03', # Reassociation response
              '04', # Probe request
              '05', # Probe response
              '08', # Beacon
              '09', # ATIM
              '0a', # Disassociation
              '0b', # Authentication
              '0c', # Deauthentication
              '0d', # Action
              '18', # BlockAckReq
              '19', # BlockAck
              '1a', # PS-Poll
              '1b', # RTS
              '1c', # CTS
              '1d', # Ack
              '1e', # CF-end
              '1f', # CF-end + CF-ack
              '20', # Data
              '21', # Data + CF-ack
              '22', # Data + CF-poll
              '23', # Data + CF-ack + CF-poll
              '24', # Null
              '25', # CF-ack
              '26', # CF-poll
              '27', # CF-ack + CF-poll
              '28', # QoS data
              '29', # QoS data + CF-ack
              '2a', # QoS data + CF-poll
              '2b', # QoS data + CF-ack + CF-poll
              '2c', # QoS null
              '2e', # QoS + CF-poll
              '2f'] # QoS + CF-ack

# Convert lines of frame to Python dictionaries
def sniffer_log_to_dict(sniffer_log: "List[Str]") -> "List[Dict]":
    ret_dicts = []
    last_timestamp = 0
    overflow_counts = 0
    for l in sniffer_log:
        ret_dict = {}
        try:
            this_timestamp = int(l[:10])/1e6
            ret_dict["timestamp"] = this_timestamp
            ret_dict["channel"] = int(l[13:15])
            ret_dict["rssi"] = int(l[18:21])
        except:
            # Corrupted
            continue
        ret_dict["frame_type"] = l[22:24]
        if ret_dict["frame_type"] not in validtypes:
            # Invalid type received, skip it
            continue
        ret_dict["ds_flag"] = l[25:27]
        ret_dict["dest_mac"] = l[28:40]
        ret_dict["src_mac"] = l[41:53]
        ret_dict["addr3"] = l[54:66]
        ret_dict["addr4"] = l[67:79]
        ret_dicts += [ret_dict]
    return ret_dicts

def plot_sniffed_frames(path):
    sniffed_frames = []
    with open(path) as f:
        for l in f:
            if re.match('^.*?\d{10} CH\d{2} RI-\d+ \d{2} \d{2}',l):
                frame_line = l[21:]
                sniffed_frames += [frame_line]
    sniffed_frames = sniffer_log_to_dict(sniffed_frames)
    plt.figure(figsize=(16,8))
    for i in range(1,14):
        sniffed_frames_filtered = [f for f in sniffed_frames if f["channel"] == i]
        plt.scatter([d["timestamp"] for d in sniffed_frames_filtered],[d["rssi"] for d in sniffed_frames_filtered],label="Channel %d"%i)
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("RSSI")
    plt.ylim(-100,0)
    plt.title("Frames from different channels")
    plt.savefig("smart_hopping.png")
    plt.show()

plot_sniffed_frames("smart_hopping_chiwah_outsideroom.log")
fig = plt.gcf()

hopper_msg_list = load_time_series("smart_hopping_outdoors_shielded.log")
hopper_msg_list = filter_time_series_hopper(hopper_msg_list)
hopper_msg_list = calculate_hopper_duration_percent(hopper_msg_list)
plot_time_series_hopper(hopper_msg_list,"Shielded room -> Outdoors")
plot_sniffed_frames("smart_hopping_outdoors_shielded.log")



