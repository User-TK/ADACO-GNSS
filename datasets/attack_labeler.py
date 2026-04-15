# datasets/attack_labeler.py
# 3-class label logic for ADACO-GNSS project.
#
# Uses per-burst timestamps from Table 13 of:
#   Wang et al., "GNSS interference and spoofing dataset"
#   Data in Brief 54 (2024) 110302
#
# My previous version labeled entire hour blocks as attacked, which mislabeled
# the large gaps between individual bursts as spoofing/jamming.
#
# All intervals are stored as absolute seconds from midnight:
#   abs_sec = hour * 3600 + minute * 60 + second
# This handles the two bursts that cross an hour boundary cleanly
# (spoofing attack 4: 12:57:00-13:02:00, spoofing attack 19: 15:55:00-16:00:00).
#
# Jamming is checked before spoofing because the jamming window (16:56+)
# begins before the spoofing window fully ends (16:44).

import numpy as np

# fmt: off
_SPOOF_INTERVALS = [
    # (start_abs_sec, end_abs_sec) inclusive
    (12*3600 + 32*60 + 30,  12*3600 + 37*60 +  0),  # Attack 1
    (12*3600 + 38*60 + 41,  12*3600 + 41*60 + 20),  #  and so on
    (12*3600 + 44*60 +  5,  12*3600 + 55*60 + 50),  # 
    (12*3600 + 57*60 +  0,  13*3600 +  2*60 +  0),  # (this one spans hour 12->13) 
    (13*3600 + 18*60 + 30,  13*3600 + 23*60 + 40),  # 
    (13*3600 + 27*60 + 50,  13*3600 + 37*60 + 50),  # 
    (13*3600 + 37*60 +  0,  13*3600 + 46*60 +  0),  # 
    (13*3600 + 47*60 + 10,  13*3600 + 56*60 +  0),  # 
    (13*3600 + 56*60 +  5,  13*3600 + 59*60 + 59),  # 
    (14*3600 +  5*60 +  0,  14*3600 + 10*60 + 18),  # 
    (14*3600 + 12*60 +  0,  14*3600 + 15*60 +  0),  # 
    (14*3600 + 18*60 +  0,  14*3600 + 30*60 +  0),  # 
    (14*3600 + 35*60 +  0,  14*3600 + 40*60 + 15),  # 
    (14*3600 + 45*60 +  0,  14*3600 + 50*60 + 30),  # 
    (14*3600 + 52*60 +  0,  14*3600 + 58*60 +  0),  #
    (15*3600 + 11*60 +  0,  15*3600 + 15*60 +  0),  # 
    (15*3600 + 17*60 +  0,  15*3600 + 30*60 +  0),  # 
    (15*3600 + 43*60 +  0,  15*3600 + 48*60 +  0),  # 
    (15*3600 + 55*60 +  0,  16*3600 +  0*60 +  0),  # (spans hour 15->16)
    (16*3600 + 10*60 +  0,  16*3600 + 15*60 + 30),  # 
    (16*3600 + 20*60 +  0,  16*3600 + 23*60 +  0),  # 
    (16*3600 + 30*60 +  0,  16*3600 + 36*60 +  0),  # 
    (16*3600 + 40*60 +  0,  16*3600 + 44*60 +  0),  # Attack 23
]

_JAM_INTERVALS = [
    # (start_abs_sec, end_abs_sec) inclusive
    (16*3600 + 56*60 +  0,  16*3600 + 57*60 +  0),  # Attack 1
    (16*3600 + 58*60 +  3,  16*3600 + 59*60 +  3),  
    (17*3600 +  0*60 + 20,  17*3600 +  1*60 + 26),  # and so on
    (17*3600 +  3*60 +  0,  17*3600 +  4*60 +  0),  
    (17*3600 +  6*60 +  0,  17*3600 +  7*60 +  0),  
    (17*3600 +  9*60 +  0,  17*3600 + 10*60 +  0),  
    (17*3600 + 12*60 +  0,  17*3600 + 13*60 +  0),  
    (17*3600 + 15*60 +  0,  17*3600 + 16*60 +  0),  
    (17*3600 + 17*60 + 30,  17*3600 + 18*60 + 30),  # finnaly
    (17*3600 + 19*60 + 30,  17*3600 + 20*60 + 30),  # Attack numero 10
]

# fmt: on


def compute_attack_type_from_time(day, hour, second_within_hour):
    """
    Returns the 3-class label for a single sample.

    Labels:
        0 -- clean
        1 -- spoofing
        2 -- jamming

    Parameters
    ----------
    day : bytes
        HDF5 day key, e.g. b"1221" or b"12".
    hour : int
        Hour of day 0-23.
    second_within_hour : int
        Row position within its (day, hour) block, 0-indexed.
        Maps to clock second HH:MM:SS where MM*60+SS == second_within_hour
        because the dataset has 1 Hz cadence and continuous 3600-file hours.
    """
    if day != b"1221":
        return 0

    abs_sec = hour * 3600 + second_within_hour

    for start, end in _JAM_INTERVALS:
        if start <= abs_sec <= end:
            return 2

    for start, end in _SPOOF_INTERVALS:
        if start <= abs_sec <= end:
            return 1

    return 0


def compute_labels_batch(days, hours, seconds):
    """
    Vectorized version of compute_attack_type_from_time for use on full arrays.
    Avoids a Python loop over 1.6M samples.

    Parameters
    ----------
    days : np.ndarray of bytes
        Shape (N,). Values like b"1221", b"12", etc.
    hours : np.ndarray of int
        Shape (N,). Hour of day 0-23.
    seconds : np.ndarray of int
        Shape (N,). second_within_hour, 0-indexed per (day, hour) block.

    Returns
    -------
    labels : np.ndarray of int32, shape (N,)
        0 = clean, 1 = spoofing, 2 = jamming.
    """
    labels = np.zeros(len(days), dtype=np.int32)

    mask_1221 = days == b"1221"
    if not mask_1221.any():
        return labels

    idx = np.where(mask_1221)[0]
    abs_secs = hours[idx].astype(np.int32) * 3600 + seconds[idx].astype(np.int32)

    # Jamming first -- priority over spoofing at the 16:44-16:56 boundary
    for start, end in _JAM_INTERVALS:
        hit = (abs_secs >= start) & (abs_secs <= end)
        labels[idx[hit]] = 2

    # Spoofing only where not already marked as jamming
    not_jam = labels[idx] != 2
    for start, end in _SPOOF_INTERVALS:
        hit = (abs_secs >= start) & (abs_secs <= end) & not_jam
        labels[idx[hit]] = 1
        # update not_jam incrementally so overlapping spoof bursts don't clobber
        not_jam = labels[idx] != 2

    return labels