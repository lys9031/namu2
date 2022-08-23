import streamlit as st
import multiprocessing as mp
import configparser
import cv2
import matplotlib.pyplot as plt
import SessionState

import sys
import numpy as np
import pandas as pd
import argparse
import glob
import os

# from func_factory import *
# from color_module import color_info

# from demo.demo import setup_cfg, get_parser
# from demo.predictor import VisualizationDemo
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from PIL import Image

from integrationSystem import result
from integrationSystem import sorting


@st.cache(suppress_st_warning=True)
def imgload(filepath, dataselect):
    src = cv2.imread(filepath)
    print("\n")
    print(dataselect)
    print("original image shape = {0}".format(src.shape))

    session_state.detection = False
    session_state.crop = False
    session_state.color = False

    return dataselect, src


session_state = SessionState.get(submit=False, crop=False, detection=False, color=False)
input_dirpath = st.sidebar.text_input("Enter the directory path", "sample")
if os.path.exists(input_dirpath) == False:
    session_state.submit = False
    st.sidebar.warning('Please check directory path')

elif st.sidebar.button("Submit", key='input_dirpath'):
    session_state.submit = True

if session_state.submit:
    dirpath = input_dirpath
    fileids = glob.glob(os.path.join(dirpath, '*.jpg')) + glob.glob(os.path.join(dirpath, '*.jpeg')) + glob.glob(
        os.path.join(dirpath, '*.png'))
    fileidlist = []
    for fileid in fileids:
        if (os.path.basename(fileid)[-4:] == 'jpeg'):
            fileidlist.append(os.path.basename(fileid))
        else:
            fileidlist.append(os.path.basename(fileid))
    selectlist = ['Select picture'] + fileidlist

    ## Select Box
    session_state.dataselect = st.sidebar.selectbox('Select Data', selectlist)

    if session_state.dataselect == 'Select picture':
        st.sidebar.success('To continue select picture.')
    else:
        ## Buttons
        # detection = st.sidebar.button('detection')
        # color = st.sidebar.button('color histogram')

        idx = selectlist.index(session_state.dataselect) - 1
        filename, session_state.src = imgload(fileids[idx], session_state.dataselect)
        session_state.src = read_image(fileids[idx], format="BGR")

        ## Show image
        st.markdown('## 원본 그림')
        img = cv2.cvtColor(session_state.src, cv2.COLOR_BGR2RGB)
        st.image(img, width=500, caption=session_state.dataselect)
        ## Slider
        dist_score = st.slider("distance score", 1, 100, 80)
        size_score = st.slider("size score", 1, 100, 80)
        histogram_score = st.slider("histogram score", 1, 100, 20)
        triplet_score = st.slider("triplet score", 1, 100, 50)

        # st.write(dist_score, size_score, histogram_score, triplet_score)

        crop = st.button('AI 분석')

        # dist_score = 80
        # size_score = 80
        # histogram_score = 20
        # triplet_score = 50
        ii=0

        

        if crop:

            scoring_result = result(fileids[idx], dist_score, size_score, histogram_score, triplet_score)
            csv_result = pd.read_csv('./csvFile/final_integrate.csv', nrows=4)
            # print("scoring_result : ", scoring_result)
            # print("csv_result : ", csv_result)
            st.write('인공지능이 찾은 내 아이의 그림과 비슷한 그림입니다')


            for re in scoring_result[:4]:
                st.write('그림과', csv_result['score'][ii], '% 유사한 그림입니다')
                # print(type(csv_result['score'][ii]))
                # print("re : ", re[0])
                session_state.src = read_image(re, format="BGR")
                img = cv2.cvtColor(session_state.src, cv2.COLOR_BGR2RGB)               
                st.image(img, width=500, caption=session_state.dataselect)
                ii=ii+1
