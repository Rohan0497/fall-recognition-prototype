"""Streamlit demo: upload video -> run inference -> show class probabilities."""

import streamlit as st

st.set_page_config(page_title='Fall Recognition Demo', layout='centered')

st.title('Fall Recognition Demo')

uploaded = st.file_uploader('Upload a video file', type=['mp4', 'avi', 'mov', 'mkv'])

if uploaded is not None:
    st.video(uploaded)

    if st.button('Run prediction'):
        # TODO: load model, run preprocessing + (optional) OpenPose step, return probabilities
        st.info('TODO: implement inference. Show probabilities here.')
