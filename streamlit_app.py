import os, sys
import cv2, streamlit as st
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE + "/src")
st.title("Deploy test")
import inference
img = cv2.cvtColor(cv2.imread(BASE + "/examples/01_whq.jpg"), cv2.COLOR_BGR2RGB)
ann, msg, scores, zw = inference.predict_gfe(img)
st.image(ann)
st.write(msg)
st.write(scores)
