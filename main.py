import streamlit as st
import pandas as pd
import base64
import cv2 as cv
import numpy as np
import av
import mediapipe as mp
import os
import tempfile
from cvzone.HandTrackingModule import HandDetector
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from glob import glob
from tensorflow.keras.models import load_model
from typing import Dict, List, Optional, Union



def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
        unsafe_allow_html=True
    )


add_bg_from_local('bg_5hands.jpg')
streamlit_style = """
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100&display=swap');
            html, body, [class*="css"]  {
            font-family: 'Roboto', sans-serif;
            }
            </style>
            """
st.markdown(streamlit_style, unsafe_allow_html=True)

original_title = '<p style="font-family:Optima; color:Orange; font-size: 40px;text-align:left top">Welcome to NyoKi Classifier</p>'
st.markdown(original_title, unsafe_allow_html=True)
#st.header(':gray[_Welcome to NyoKi Classifier_]')
#st.subheader('Hand Sign Recognition Application (Word Level)')
activities = ["Home", "Webcam Hand Detection"]
choice_s = st.sidebar.selectbox("Select Activity <3", activities)




mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
mp_hands = mp.solutions.hands

model = load_model('models for streamlit/action_yangon_100acc_100val')

def process(image,sequence=[], sentence=[], predictions=[], thereshold = 0.5):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:

        words = np.array(['[MingalarPar]', '[Mhan]', '[Mharr]'])

        img_resize = cv.resize(image, (640, 480))
        # Make detections
        image_detect, results = mediapipe_detection(img_resize, hands)

        if results.multi_hand_landmarks:
            # Draw landmarks
            draw_landmarks(image_detect, results)

            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-90:]

            if len(sequence) == 90:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                #             print(words[np.argmax(res)])
                predictions.append(np.argmax(res))

                # 3. Viz logic
                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > thereshold:

                        if len(sentence) > 0:
                            if words[np.argmax(res)] != sentence[-1]:
                                sentence.append(words[np.argmax(res)])
                        else:
                            sentence.append(words[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # Viz probabilities
                image = prob_viz(res, words, image_detect, colors)
        return image


colors = [(255,255,255), (71,240,21), (26,21,235)]

def mediapipe_detection(image,hands):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = hands.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results

def prob_viz(res, words, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv.rectangle(output_frame, (0, 2 + num * 40), (int(prob * 155), 32 + num * 40), colors[num], -1)
        cv.putText(output_frame, words[num], (0, 25 + num * 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                   cv.LINE_AA)

    return output_frame

def extract_keypoints(results):
    keypoints = []
    if results.multi_hand_landmarks:
        for h_lmk in results.multi_hand_landmarks[0].landmark:
            keypoints.append(np.array([h_lmk.x, h_lmk.y, h_lmk.z]))
    else:
        keypoints.append(np.zeros(21 * 3))

    keypoints = np.array(keypoints).flatten()
    return keypoints

def draw_landmarks(image, results):
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img = process(img)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


if choice_s == "Home":
    html_temp_home1 = """<div style="width:550px;height:100px;background-color:#BCB6B5;background-position:center;padding:10px">
                                  <h4 style="color:black;text-align:center;">
                                  Hand Sign (word level) recognition application<br>using OpenCV, Streamlit.
                                  </h4>
                                  </div>
                                  </br>"""
    st.markdown(html_temp_home1, unsafe_allow_html=True)
    # WhyWeMakeThisProject
    original_subheader = '<p style="color:LightSteelBlue; font-size: 30px;">Why we make this project!</p>'
    st.markdown(original_subheader, unsafe_allow_html=True)
    # MotivationText
    motivation_text = """<p style="color:SkyBlue;"><b>In reality, everyone is not perfect and unfortunately, some are even born with disabilities. Their lives are unfair from the start of their chapter. Realizing that, we got the idea of creating an app for the deaf people so that they can equally
         enjoy their social lives alongside everyone else around the world.</b></p>
        """
    st.markdown(motivation_text, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("မင်္ဂလာပါ")
        st.video("main_page_videos/3.mp4")

    with col2:
        st.header("မှန်")
        st.video("main_page_videos/2.mp4")

    with col3:
        st.header("မှား")
        st.video("main_page_videos/1.mp4")




elif choice_s == "Webcam Hand Detection":
    webcam_subheader = '<p style="font-family:Monospace; color:SkyBlue; font-size: 15px;">1st - Choose your webcam device.<br>2nd - Press start to begin hand detection.</p>'
    st.markdown(webcam_subheader, unsafe_allow_html=True)
    webcam_secondheader = '<p style="font-family:Monospace; color:LightSteelBlue; font-size: 10px;">Exception : It will take one or two seconds before it starts detection.</p>'
    st.markdown(webcam_secondheader, unsafe_allow_html=True)
    
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [
            {
                "urls": ["turn:numb.viagenie.ca"],
                "username": ["webrtc@live.com"],
                "credential": ["muazkh"]
            }
        ]
        }
    )
    webrtc_ctx = webrtc_streamer(
        key="WYH",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=VideoProcessor,
        async_processing=True,
    )

