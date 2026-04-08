"""

Please place this script into the folder "./GNSS dataset/"
You can use this code to extract and integrate observations,
satellites information and PVT solutions from the raw data.

The contributors:
Xiaoyan Wang (xiaoyan_wang2020@163.com)
Jingjing Yang (yangjingjing@ynu.edu.cn)
Ming Huang (huangming@ynu.edu.cn)

12/28 2023

"""

import json, os
import numpy as np


def main_code_rax(days, hours):
    """

    This code is used to extract multi-GNSS observations, such as pseudorange, Doppler,
    carrier phase and signal quality information for navigation satellites from RXM-RAWX
    and save as observationHH.json.


    :param days: describes the date, which corresponds to
                 the next level of the processed data folder, with options
                 ranging from September 12th to 30th and December 21st.
    :param hours: describes the hour, ranging from 0 to 23
    :return: None
    """

    def read_single_rax(content, numSats, gnssid, sigId2):
        """
        This code is used to read a single file (e.g. "Raw data/12/0/RXM-RAWX/2023-09-12 00-00-01.json")
        and extract the observations of all visible satellites within it.

        :param content: json file content opened by python
        :param numSats: number of satellites in different constellation (see Table 2) GPS:32, Galileo:36, BDS:63, QZSS:10, GLONASS: 32
        :param gnssid: ID of GNSS (see Table 2), GPS:0, Galileo:2, BDS:3, QZSS:5, GLONASS: 6
        :param sigId2: ID of signal (see Table 2), GPS:[0,3], Galileo:[0,6], BDS:[0,2], QZSS:[0,5], GLONASS:[0,2]
        :return: observations lists
        """
        numMeas = content['numMeas']
        VS = np.zeros((1, numSats))  # receive the visible satellites
        # observation metrix
        cno_G_L1CA, cno_G_L2C = np.zeros((1, numSats)), np.zeros((1, numSats))  # CN0
        prMes_G_L1CA, prMes_G_L2C = np.zeros((1, numSats)), np.zeros((1, numSats))  # prMes
        cpMes_G_L1CA, cpMes_G_L2C = np.zeros((1, numSats)), np.zeros((1, numSats))  # cpMes
        doMes_G_L1CA, doMes_G_L2C = np.zeros((1, numSats)), np.zeros((1, numSats))  # doMes
        prStd_G_L1CA, prStd_G_L2C = np.zeros((1, numSats)), np.zeros((1, numSats))  # prStd
        cpStd_G_L1CA, cpStd_G_L2C = np.zeros((1, numSats)), np.zeros((1, numSats))  # cpStd
        doStd_G_L1CA, doStd_G_L2C = np.zeros((1, numSats)), np.zeros((1, numSats))  # doStd

        # Iterate over the measurements (1-9):
        for i in range(1, 10):
            # select the target constellation:
            if content['gnssId_0%s' % i] == gnssid:
                isvId = int(content['svId_0%s' % i] - 1)
                if isvId > numSats:
                    isvId = -1
                VS[0, isvId] = content['svId_0%s' % i]
                # select the first signal band:
                if content['sigId_0%s' % i] == 0:
                    cno_G_L1CA[0, isvId] = content['cno_0%s' % i]
                    prMes_G_L1CA[0, isvId] = content['prMes_0%s' % i]
                    cpMes_G_L1CA[0, isvId] = content['cpMes_0%s' % i]
                    doMes_G_L1CA[0, isvId] = content['doMes_0%s' % i]
                    prStd_G_L1CA[0, isvId] = content['prStd_0%s' % i]
                    cpStd_G_L1CA[0, isvId] = content['cpStd_0%s' % i]
                    doStd_G_L1CA[0, isvId] = content['doStd_0%s' % i]
                # select the 2nd signal band:
                elif content['sigId_0%s' % i] == sigId2:
                    cno_G_L2C[0, isvId] = content['cno_0%s' % i]
                    prMes_G_L2C[0, isvId] = content['prMes_0%s' % i]
                    cpMes_G_L2C[0, isvId] = content['cpMes_0%s' % i]
                    doMes_G_L2C[0, isvId] = content['doMes_0%s' % i]
                    prStd_G_L2C[0, isvId] = content['prStd_0%s' % i]
                    cpStd_G_L2C[0, isvId] = content['cpStd_0%s' % i]
                    doStd_G_L2C[0, isvId] = content['doStd_0%s' % i]
        # Iterate over the measurements (10-):
        for i in range(10, numMeas + 1):
            # select the target constellation:
            if content['gnssId_%s' % i] == gnssid:
                isvId = int(content['svId_%s' % i] - 1)
                if isvId > numSats:
                    isvId = -1
                VS[0, isvId] = content['svId_%s' % i]
                # select the first signal band:
                if content['sigId_%s' % i] == 0:  # sigId = 0
                    cno_G_L1CA[0, isvId] = content['cno_%s' % i]
                    prMes_G_L1CA[0, isvId] = content['prMes_%s' % i]
                    cpMes_G_L1CA[0, isvId] = content['cpMes_%s' % i]
                    doMes_G_L1CA[0, isvId] = content['doMes_%s' % i]
                    prStd_G_L1CA[0, isvId] = content['prStd_%s' % i]
                    cpStd_G_L1CA[0, isvId] = content['cpStd_%s' % i]
                    doStd_G_L1CA[0, isvId] = content['doStd_%s' % i]
                # select the 2nd signal band:
                elif content['sigId_%s' % i] == sigId2:
                    cno_G_L2C[0, isvId] = content['cno_%s' % i]
                    prMes_G_L2C[0, isvId] = content['prMes_%s' % i]
                    cpMes_G_L2C[0, isvId] = content['cpMes_%s' % i]
                    doMes_G_L2C[0, isvId] = content['doMes_%s' % i]
                    prStd_G_L2C[0, isvId] = content['prStd_%s' % i]
                    cpStd_G_L2C[0, isvId] = content['cpStd_%s' % i]
                    doStd_G_L2C[0, isvId] = content['doStd_%s' % i]
        # return the extracted observation metrix and visiblie satellites
        return np.ndarray.tolist(cno_G_L1CA), np.ndarray.tolist(cno_G_L2C), \
            np.ndarray.tolist(prMes_G_L1CA), np.ndarray.tolist(prMes_G_L2C), \
            np.ndarray.tolist(cpMes_G_L1CA), np.ndarray.tolist(cpMes_G_L2C), \
            np.ndarray.tolist(doMes_G_L1CA), np.ndarray.tolist(doMes_G_L2C), \
            np.ndarray.tolist(prStd_G_L1CA), np.ndarray.tolist(prStd_G_L2C), \
            np.ndarray.tolist(cpStd_G_L1CA), np.ndarray.tolist(cpStd_G_L2C), \
            np.ndarray.tolist(doStd_G_L1CA), np.ndarray.tolist(doStd_G_L2C), \
            np.ndarray.tolist(VS)
    # provide days and hours:
    for day in days:
        for hour in hours:
            recordTime = []  # receiver the received time
            json_name = 'raw/%s/%s/RXM-RAWX' % (day, hour)
            fileName = os.listdir(json_name)  # get all json files from the data path

            # receive the visible satellites of different constellations
            VSG, VSE, VSB, VSQ, VSR = [], [], [], [], []
            # receive the observations of different constellation
            prMes_G1, doMes_G1, cpMes_G1, cn0_G1, prStd_G1, cpStd_G1, doStd_G1 = [], [], [], [], [], [], []
            prMes_G2, doMes_G2, cpMes_G2, cn0_G2, prStd_G2, cpStd_G2, doStd_G2 = [], [], [], [], [], [], []

            prMes_E1, doMes_E1, cpMes_E1, cn0_E1, prStd_E1, cpStd_E1, doStd_E1 = [], [], [], [], [], [], []
            prMes_E2, doMes_E2, cpMes_E2, cn0_E2, prStd_E2, cpStd_E2, doStd_E2 = [], [], [], [], [], [], []

            prMes_B1, doMes_B1, cpMes_B1, cn0_B1, prStd_B1, cpStd_B1, doStd_B1 = [], [], [], [], [], [], []
            prMes_B2, doMes_B2, cpMes_B2, cn0_B2, prStd_B2, cpStd_B2, doStd_B2 = [], [], [], [], [], [], []

            prMes_Q1, doMes_Q1, cpMes_Q1, cn0_Q1, prStd_Q1, cpStd_Q1, doStd_Q1 = [], [], [], [], [], [], []
            prMes_Q2, doMes_Q2, cpMes_Q2, cn0_Q2, prStd_Q2, cpStd_Q2, doStd_Q2 = [], [], [], [], [], [], []

            prMes_R1, doMes_R1, cpMes_R1, cn0_R1, prStd_R1, cpStd_R1, doStd_R1 = [], [], [], [], [], [], []
            prMes_R2, doMes_R2, cpMes_R2, cn0_R2, prStd_R2, cpStd_R2, doStd_R2 = [], [], [], [], [], [], []

            for file_name in fileName:
                print(os.path.join(json_name, file_name))
                with open(os.path.join(json_name, file_name), "r", encoding="utf-8") as f:
                    file_content = json.load(f)
                recordTime.append(file_content['start_time'])
                gnssSats = [32, 36, 63, 10, 33]  # number of satellites in different constellations
                gnssIds = [0, 2, 3, 5, 6]
                sigId2s = [3, 6, 2, 5, 2]
                for i1 in range(len(gnssIds)):
                    gnssId = gnssIds[i1]
                    numSats = gnssSats[i1]
                    sigId2 = sigId2s[i1]
                    # print('gnssId=',gnssId, ', numSats=',numSats, ', sigId2=',sigId2)
                    cu_tuple = read_single_rax(file_content, numSats, gnssId, sigId2)
                    if gnssId == 0:  # GPS

                        cn0_G1, cn0_G2 = cn0_G1 + cu_tuple[0], cn0_G2 + cu_tuple[1]
                        prMes_G1, prMes_G2 = prMes_G1 + cu_tuple[2], prMes_G2 + cu_tuple[3]
                        cpMes_G1, cpMes_G2 = cpMes_G1 + cu_tuple[4], cpMes_G2 + cu_tuple[5]
                        doMes_G1, doMes_G2 = doMes_G1 + cu_tuple[6], doMes_G2 + cu_tuple[7]
                        prStd_G1, prStd_G2 = prStd_G1 + cu_tuple[8], prStd_G2 + cu_tuple[9]
                        cpStd_G1, cpStd_G2 = cpStd_G1 + cu_tuple[10], cpStd_G2 + cu_tuple[11]
                        doStd_G1, doStd_G2 = doStd_G1 + cu_tuple[12], doStd_G2 + cu_tuple[13]
                        VSG = VSG + cu_tuple[-1]

                    elif gnssId == 2:  # Galileo
                        cn0_E1, cn0_E2 = cn0_E1 + cu_tuple[0], cn0_E2 + cu_tuple[1]
                        prMes_E1, prMes_E2 = prMes_E1 + cu_tuple[2], prMes_E2 + cu_tuple[3]
                        cpMes_E1, cpMes_E2 = cpMes_E1 + cu_tuple[4], cpMes_E2 + cu_tuple[5]
                        doMes_E1, doMes_E2 = doMes_E1 + cu_tuple[6], doMes_E2 + cu_tuple[7]
                        prStd_E1, prStd_E2 = prStd_E1 + cu_tuple[8], prStd_E2 + cu_tuple[9]
                        cpStd_E1, cpStd_E2 = cpStd_E1 + cu_tuple[10], cpStd_E2 + cu_tuple[11]
                        doStd_E1, doStd_E2 = doStd_E1 + cu_tuple[12], doStd_E2 + cu_tuple[13]
                        VSE = VSE + cu_tuple[-1]

                    elif gnssId == 3:  # BDS
                        cn0_B1, cn0_B2 = cn0_B1 + cu_tuple[0], cn0_B2 + cu_tuple[1]
                        prMes_B1, prMes_B2 = prMes_B1 + cu_tuple[2], prMes_B2 + cu_tuple[3]
                        cpMes_B1, cpMes_B2 = cpMes_B1 + cu_tuple[4], cpMes_B2 + cu_tuple[5]
                        doMes_B1, doMes_B2 = doMes_B1 + cu_tuple[6], doMes_B2 + cu_tuple[7]
                        prStd_B1, prStd_B2 = prStd_B1 + cu_tuple[8], prStd_B2 + cu_tuple[9]
                        cpStd_B1, cpStd_B2 = cpStd_B1 + cu_tuple[10], cpStd_B2 + cu_tuple[11]
                        doStd_B1, doStd_B2 = doStd_B1 + cu_tuple[12], doStd_B2 + cu_tuple[13]
                        VSB = VSB + cu_tuple[-1]

                    elif gnssId == 5:  # QZSS
                        cn0_Q1, cn0_Q2 = cn0_Q1 + cu_tuple[0], cn0_Q2 + cu_tuple[1]
                        prMes_Q1, prMes_Q2 = prMes_Q1 + cu_tuple[2], prMes_Q2 + cu_tuple[3]
                        cpMes_Q1, cpMes_Q2 = cpMes_Q1 + cu_tuple[4], cpMes_Q2 + cu_tuple[5]
                        doMes_Q1, doMes_Q2 = doMes_Q1 + cu_tuple[6], doMes_Q2 + cu_tuple[7]
                        prStd_Q1, prStd_Q2 = prStd_Q1 + cu_tuple[8], prStd_Q2 + cu_tuple[9]
                        cpStd_Q1, cpStd_Q2 = cpStd_Q1 + cu_tuple[10], cpStd_Q2 + cu_tuple[11]
                        doStd_Q1, doStd_Q2 = doStd_Q1 + cu_tuple[12], doStd_Q2 + cu_tuple[13]
                        VSQ = VSQ + cu_tuple[-1]

                    elif gnssId == 6:  # GLONASS
                        cn0_R1, cn0_R2 = cn0_R1 + cu_tuple[0], cn0_R2 + cu_tuple[1]
                        prMes_R1, prMes_R2 = prMes_R1 + cu_tuple[2], prMes_R2 + cu_tuple[3]
                        cpMes_R1, cpMes_R2 = cpMes_R1 + cu_tuple[4], cpMes_R2 + cu_tuple[5]
                        doMes_R1, doMes_R2 = doMes_R1 + cu_tuple[6], doMes_R2 + cu_tuple[7]
                        prStd_R1, prStd_R2 = prStd_R1 + cu_tuple[8], prStd_R2 + cu_tuple[9]
                        cpStd_R1, cpStd_R2 = cpStd_R1 + cu_tuple[10], cpStd_R2 + cu_tuple[11]
                        doStd_R1, doStd_R2 = doStd_R1 + cu_tuple[12], doStd_R2 + cu_tuple[13]
                        VSR = VSR + cu_tuple[-1]

            #  keys of the dictionary
            keys2 = ['recordTime', 'VSG', 'VSE', 'VSB', 'VSQ', 'VSR',
                     'prMes_G1', 'doMes_G1', 'cpMes_G1', 'cn0_G1', 'prStd_G1', 'cpStd_G1', 'doStd_G1',
                     'prMes_G2', 'doMes_G2', 'cpMes_G2', 'cn0_G2', 'prStd_G2', 'cpStd_G2', 'doStd_G2',
                     'prMes_E1', 'doMes_E1', 'cpMes_E1', 'cn0_E1', 'prStd_E1', 'cpStd_E1', 'doStd_E1',
                     'prMes_E2', 'doMes_E2', 'cpMes_E2', 'cn0_E2', 'prStd_E2', 'cpStd_E2', 'doStd_E2',
                     'prMes_B1', 'doMes_B1', 'cpMes_B1', 'cn0_B1', 'prStd_B1', 'cpStd_B1', 'doStd_B1',
                     'prMes_B2', 'doMes_B2', 'cpMes_B2', 'cn0_B2', 'prStd_B2', 'cpStd_B2', 'doStd_B2',
                     'prMes_Q1', 'doMes_Q1', 'cpMes_Q1', 'cn0_Q1', 'prStd_Q1', 'cpStd_Q1', 'doStd_Q1',
                     'prMes_Q2', 'doMes_Q2', 'cpMes_Q2', 'cn0_Q2', 'prStd_Q2', 'cpStd_Q2', 'doStd_Q2',
                     'prMes_R1', 'doMes_R1', 'cpMes_R1', 'cn0_R1', 'prStd_R1', 'cpStd_R1', 'doStd_R1',
                     'prMes_R2', 'doMes_R2', 'cpMes_R2', 'cn0_R2', 'prStd_R2', 'cpStd_R2', 'doStd_R2']
            #  values of the dictionary
            values2 = [recordTime, VSG, VSE, VSB, VSQ, VSR,
                       prMes_G1, doMes_G1, cpMes_G1, cn0_G1, prStd_G1, cpStd_G1, doStd_G1,
                       prMes_G2, doMes_G2, cpMes_G2, cn0_G2, prStd_G2, cpStd_G2, doStd_G2,
                       prMes_E1, doMes_E1, cpMes_E1, cn0_E1, prStd_E1, cpStd_E1, doStd_E1,
                       prMes_E2, doMes_E2, cpMes_E2, cn0_E2, prStd_E2, cpStd_E2, doStd_E2,
                       prMes_B1, doMes_B1, cpMes_B1, cn0_B1, prStd_B1, cpStd_B1, doStd_B1,
                       prMes_B2, doMes_B2, cpMes_B2, cn0_B2, prStd_B2, cpStd_B2, doStd_B2,
                       prMes_Q1, doMes_Q1, cpMes_Q1, cn0_Q1, prStd_Q1, cpStd_Q1, doStd_Q1,
                       prMes_Q2, doMes_Q2, cpMes_Q2, cn0_Q2, prStd_Q2, cpStd_Q2, doStd_Q2,
                       prMes_R1, doMes_R1, cpMes_R1, cn0_R1, prStd_R1, cpStd_R1, doStd_R1,
                       prMes_R2, doMes_R2, cpMes_R2, cn0_R2, prStd_R2, cpStd_R2, doStd_R2]

            # format all observations as a dictionary type and save it in the specified path.
            save_dict2 = dict(zip(keys2, values2))
            new_data2 = json.loads(str(save_dict2).replace("'", "\""))
            savePath = 'processed data/%s/' % (day)
            if os.path.exists(savePath) == False:
                os.mkdir(savePath)
            with open(savePath + 'observation%s.json' % (hour), 'w', encoding='utf8') as f3:
                json.dump(new_data2, f3, ensure_ascii=False, indent=2)


def main_code_sat(days, hours):
    """

       This code is used to extract multi-GNSS satellites information, such as svId, svUsed,
       elev,azim from NAV-SAT and save as satelliteInfomation.json.


       :param days: describes the date, which corresponds to
                    the next level of the processed data folder, with options
                    ranging from September 12th to 30th and December 21st.
       :param hours: describes the hour, ranging from 0 to 23
       :return: None
       """

    def read_single_sat(content, numSats, gnssid):
        """
            This code is used to read a single file (e.g. "Raw data/12/0/NAV-SAT/2023-09-12 00-00-01.json")
            and extract the satellites information.
            :param content: json file content opened by python
            :param numSats: number of satellites in different constellation (see Table 2)
                            GPS:32, Galileo:36, BDS:63, QZSS:10, GLONASS: 32
            :param gnssid: ID of GNSS (see Table 2),
                            GPS:0, Galileo:2, BDS:3, QZSS:5, GLONASS: 6
            :return: satellites information list
        """

        numSvs = content['numSvs']  # the number of received satellites
        svId = np.zeros((1, numSats))  # the identifier of satellites
        svUsed = np.zeros((1, numSats))  # flag of the satellite being used for PVT solving
        svUsed = np.where(svUsed == 0, 0.11, svUsed)
        cno = np.zeros((1, numSats))  # C/N0
        elev = np.zeros((1, numSats))  # Elevation
        azim = np.zeros((1, numSats))  # Azimuth
        prRes = np.zeros((1, numSats))  # Pseudorange residual
        qualityInd = np.zeros((1, numSats))  # Signal quality indicator
        qualityInd = np.where(qualityInd == 0, 0.11, qualityInd)
        health = np.zeros((1, numSats))  # Signal health flag
        health = np.where(health == 0, 0.11, health)

        # Iterate over the measurements (1-9):
        for i in range(1, 10):
            if content['gnssId_0%s' % i] == gnssid:
                isvId = int(content['svId_0%s' % i] - 1)
                if isvId > numSats:
                    isvId = -1
                svId[0, isvId] = content['svId_0%s' % i]
                svUsed[0, isvId] = content['svUsed_0%s' % i]
                cno[0, isvId] = content['cno_0%s' % i]
                elev[0, isvId] = content['elev_0%s' % i]
                azim[0, isvId] = content['azim_0%s' % i]
                prRes[0, isvId] = content['prRes_0%s' % i]
                qualityInd[0, isvId] = content['qualityInd_0%s' % i]
                health[0, isvId] = content['health_0%s' % i]

        # Iterate over the measurements (10-):
        for i in range(10, numSvs + 1):
            if content['gnssId_%s' % i] == gnssid:
                isvId = int(content['svId_%s' % i] - 1)
                if isvId > numSats:
                    isvId = -1
                svId[0, isvId] = content['svId_%s' % i]
                svUsed[0, isvId] = content['svUsed_%s' % i]
                cno[0, isvId] = content['cno_%s' % i]
                elev[0, isvId] = content['elev_%s' % i]
                azim[0, isvId] = content['azim_%s' % i]
                prRes[0, isvId] = content['prRes_%s' % i]
                qualityInd[0, isvId] = content['qualityInd_%s' % i]
                health[0, isvId] = content['health_%s' % i]
        return np.ndarray.tolist(svId), \
            np.ndarray.tolist(svUsed), \
            np.ndarray.tolist(cno), \
            np.ndarray.tolist(elev), \
            np.ndarray.tolist(azim), \
            np.ndarray.tolist(prRes), \
            np.ndarray.tolist(qualityInd), np.ndarray.tolist(health)

    for day in days:
        for hour in hours:
            recordTime, numSvs = [], []
            json_name = 'G:/GNSSJson09_2/%s/%s/NAV-SAT' % (day, hour)
            fileName = os.listdir(json_name)# get all json files from the data path
            #  satellites information matrix of different constellation
            svId_G, svUsed_G, cno_G, elev_G, azim_G, prRes_G, qualityInd_G, health_G = [], [], [], [], [], [], [], []
            svId_E, svUsed_E, cno_E, elev_E, azim_E, prRes_E, qualityInd_E, health_E = [], [], [], [], [], [], [], []
            svId_B, svUsed_B, cno_B, elev_B, azim_B, prRes_B, qualityInd_B, health_B = [], [], [], [], [], [], [], []
            svId_Q, svUsed_Q, cno_Q, elev_Q, azim_Q, prRes_Q, qualityInd_Q, health_Q = [], [], [], [], [], [], [], []
            svId_R, svUsed_R, cno_R, elev_R, azim_R, prRes_R, qualityInd_R, health_R = [], [], [], [], [], [], [], []

            for file_name in fileName:
                print(os.path.join(json_name, file_name))
                with open(os.path.join(json_name, file_name), "r", encoding="utf-8") as f:
                    file_content = json.load(f)
                recordTime.append(file_content['start_time'])
                numSvs.append(file_content['numSvs'])
                dims = [32, 36, 63, 10, 33]
                gnssIds = [0, 2, 3, 5, 6]

                # receive the satellited information of different constellation:
                for i1 in range(len(gnssIds)):
                    gnssId = gnssIds[i1]
                    numSats = dims[i1]
                    cu_tuple = read_single_sat(file_content, numSats, gnssId)
                    if gnssId == 0:  # GPS
                        svId_G = svId_G + cu_tuple[0]
                        svUsed_G = svUsed_G + cu_tuple[1]
                        cno_G = cno_G + cu_tuple[2]
                        elev_G = elev_G + cu_tuple[3]
                        azim_G = azim_G + cu_tuple[4]
                        prRes_G = prRes_G + cu_tuple[5]
                        qualityInd_G = qualityInd_G + cu_tuple[6]
                        health_G = health_G + cu_tuple[-1]

                    elif gnssId == 2:  # Galileo
                        svId_E = svId_E + cu_tuple[0]
                        svUsed_E = svUsed_E + cu_tuple[1]
                        cno_E = cno_E + cu_tuple[2]
                        elev_E = elev_E + cu_tuple[3]
                        azim_E = azim_E + cu_tuple[4]
                        prRes_E = prRes_E + cu_tuple[5]
                        qualityInd_E = qualityInd_E + cu_tuple[6]
                        health_E = health_E + cu_tuple[-1]

                    elif gnssId == 3:  # BDS
                        svId_B = svId_B + cu_tuple[0]
                        svUsed_B = svUsed_B + cu_tuple[1]
                        cno_B = cno_B + cu_tuple[2]
                        elev_B = elev_B + cu_tuple[3]
                        azim_B = azim_B + cu_tuple[4]
                        prRes_B = prRes_B + cu_tuple[5]
                        qualityInd_B = qualityInd_B + cu_tuple[6]
                        health_B = health_B + cu_tuple[-1]

                    elif gnssId == 5:  # QZSS
                        svId_Q = svId_Q + cu_tuple[0]
                        svUsed_Q = svUsed_Q + cu_tuple[1]
                        cno_Q = cno_Q + cu_tuple[2]
                        elev_Q = elev_Q + cu_tuple[3]
                        azim_Q = azim_Q + cu_tuple[4]
                        prRes_Q = prRes_Q + cu_tuple[5]
                        qualityInd_Q = qualityInd_Q + cu_tuple[6]
                        health_Q = health_Q + cu_tuple[-1]

                    elif gnssId == 6:  # GLONASS
                        svId_R = svId_R + cu_tuple[0]
                        svUsed_R = svUsed_R + cu_tuple[1]
                        cno_R = cno_R + cu_tuple[2]
                        elev_R = elev_R + cu_tuple[3]
                        azim_R = azim_R + cu_tuple[4]
                        prRes_R = prRes_R + cu_tuple[5]
                        qualityInd_R = qualityInd_R + cu_tuple[6]
                        health_R = health_R + cu_tuple[-1]

            # format all values as a dictionary type and save it in the specified path.
            keys2 = ['recordTime', 'numSvs',
                     'svId_G', 'svUsed_G', 'cno_G', 'elev_G', 'azim_G', 'prRes_G', 'qualityInd_G', 'health_G',
                     'svId_E', 'svUsed_E', 'cno_E', 'elev_E', 'azim_E', 'prRes_E', 'qualityInd_E', 'health_E',
                     'svId_B', 'svUsed_B', 'cno_B', 'elev_B', 'azim_B', 'prRes_B', 'qualityInd_B', 'health_B',
                     'svId_Q', 'svUsed_Q', 'cno_Q', 'elev_Q', 'azim_Q', 'prRes_Q', 'qualityInd_Q', 'health_Q',
                     'svId_R', 'svUsed_R', 'cno_R', 'elev_R', 'azim_R', 'prRes_R', 'qualityInd_R', 'health_R']
            values2 = [recordTime, numSvs,
                       svId_G, svUsed_G, cno_G, elev_G, azim_G, prRes_G, qualityInd_G, health_G,
                       svId_E, svUsed_E, cno_E, elev_E, azim_E, prRes_E, qualityInd_E, health_E,
                       svId_B, svUsed_B, cno_B, elev_B, azim_B, prRes_B, qualityInd_B, health_B,
                       svId_Q, svUsed_Q, cno_Q, elev_Q, azim_Q, prRes_Q, qualityInd_Q, health_Q,
                       svId_R, svUsed_R, cno_R, elev_R, azim_R, prRes_R, qualityInd_R, health_R]
            save_dict2 = dict(zip(keys2, values2))
            new_data2 = json.loads(str(save_dict2).replace("'", "\""))

            savePath = 'processed data/%s/' % (day)
            if os.path.exists(savePath) == False:
                os.mkdir(savePath)
            with open(savePath + 'satelliteInfomation%s.json' % (hour), 'w', encoding='utf8') as f3:
                json.dump(new_data2, f3, ensure_ascii=False, indent=2)


def main_code_pvt(days, hours):

    """
    This code is used to extract PVT (Position, Velicity, Time) solutions and other
    related information from NAV-PVT, NAV-POSECEF, NAV-CLOCK and NAV-DOP and save
    as observationHH.json.


    :param days: describes the date, which corresponds to
                 the next level of the processed data folder, with options
                 ranging from September 12th to 30th and December 21st.
    :param hours: describes the hour, ranging from 0 to 23
    :return: None
    """

    for day in days:
        for hour in hours:
            recordTime = []
            numSV, nano, lon, lat, height, velN, velE, velD, hMSL, hAcc, vAcc, sAcc, gSpeed, headMot, headAcc = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
            ecefX, ecefY, ecefZ = [], [], []
            clkB, clkD, tAcc, fAcc = [], [], [], []
            gDOP, pDOP, tDOP, vDOP, hDOP, nDOP, eDOP = [], [], [], [], [], [], []

            json_name = 'Raw data/%s/%s/NAV-PVT' % (day, hour)
            fileName = os.listdir(json_name)  # get all json files from the data path
            for file_name in fileName:
                print(os.path.join(json_name, file_name))
                with open(os.path.join(json_name, file_name), "r", encoding="utf-8") as f:
                    PVT = json.load(f)
                recordTime.append(PVT['start_time']), numSV.append(PVT['numSV'])
                nano.append(PVT['nano'])
                lon.append(PVT['lon']), lat.append(PVT['lat']), height.append(PVT['height'])
                velN.append(PVT['velN']), velE.append(PVT['velE']), velD.append(PVT['velD'])
                hMSL.append(PVT['hMSL']), hAcc.append(PVT['hAcc']), vAcc.append(PVT['vAcc'])
                sAcc.append(PVT['sAcc']), gSpeed.append(PVT['gSpeed']), headMot.append(PVT['headMot'])
                headAcc.append(PVT['headAcc'])

            json_name = 'Raw data/%s/%s/NAV-POSECEF' % (day, hour)
            fileName = os.listdir(json_name)  # get all json files from the data path
            for file_name in fileName:
                print(os.path.join(json_name, file_name))
                with open(os.path.join(json_name, file_name), "r", encoding="utf-8") as fp:
                    poecef = json.load(fp)
                ecefX.append(poecef['ecefX'])
                ecefY.append(poecef['ecefY'])
                ecefZ.append(poecef['ecefZ'])

            json_name = 'Raw data/%s/%s/NAV-CLOCK' % (day, hour)
            fileName = os.listdir(json_name)  # get all json files from the data path
            for file_name in fileName:
                print(os.path.join(json_name, file_name))
                with open(os.path.join(json_name, file_name), "r", encoding="utf-8") as fc:
                    clock_content = json.load(fc)
                clkB.append(clock_content['clkB'])
                clkD.append(clock_content['clkD'])
                tAcc.append(clock_content['tAcc'])
                fAcc.append(clock_content['fAcc'])

            json_name = 'Raw data/%s/%s/NAV-DOP' % (day, hour)
            fileName = os.listdir(json_name)  # get all json files from the data path
            for file_name in fileName:
                print(os.path.join(json_name, file_name))
                with open(os.path.join(json_name, file_name), "r", encoding="utf-8") as fd:
                    DOP = json.load(fd)
                gDOP.append(DOP['gDOP'])
                pDOP.append(DOP['pDOP'])
                tDOP.append(DOP['tDOP'])
                vDOP.append(DOP['vDOP'])
                hDOP.append(DOP['hDOP'])
                nDOP.append(DOP['nDOP'])
                eDOP.append(DOP['eDOP'])

            # format all observations as a dictionary type and save it in the specified path.
            keys2 = ['recordTime', 'numSV', 'nano', 'lon', 'lat', 'height', 'velN', 'velE', 'velD', 'hMSL',
                     'hAcc', 'vAcc', 'sAcc', 'gSpeed', 'headMot', 'headAcc', 'ecefX', 'ecefY', 'ecefZ',
                     'clkB', 'clkD', 'tAcc', 'fAcc', 'gDOP', 'pDOP', 'tDOP', 'vDOP', 'hDOP', 'nDOP', 'eDOP']
            values2 = [recordTime, numSV, nano, lon, lat, height, velN, velE, velD, hMSL,
                       hAcc, vAcc, sAcc, gSpeed, headMot, headAcc, ecefX, ecefY, ecefZ,
                       clkB, clkD, tAcc, fAcc, gDOP, pDOP, tDOP, vDOP, hDOP, nDOP, eDOP]
            save_dict2 = dict(zip(keys2, values2))
            new_data2 = json.loads(str(save_dict2).replace("'", "\""))

            savePath = 'processed data/%s/' % (day)
            if os.path.exists(savePath) == False:
                os.mkdir(savePath)
            with open(savePath + 'pvtSolution%s.json' % (hour), 'w', encoding='utf8') as f3:
                json.dump(new_data2, f3, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    # Running these codes you will read the data for the period September 12, 14:00:00-15:59:59
    my_days = [12]  # list, the day(s) you want to be extracted, ranging from September 12th to 30th and December 21st.
    my_hours = [14]  # list, the hour(s) you want to be extracted, ranging form 0 to 23
    main_code_rax(my_days, my_hours)
    main_code_sat(my_days, my_hours)
    main_code_pvt(my_days, my_hours)

    # To  extract data for multiple consecutive time periods, add more days and hours into
    # the lists my_days1 and my_hours1. data from September 12 to 16, 00:00:00-23:59:59
    my_days1 = [12, 15,
                16]  # list, the day(s) you want to be extracted, ranging from September 12th to 30th and December 21st.
    my_hours1 = range(24)  # list, the hour(s) you want to be extracted, ranging form 0 to 23
    main_code_rax(my_days1, my_hours1)
    main_code_sat(my_days1, my_hours1)
    main_code_pvt(my_days1, my_hours1)