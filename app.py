import streamlit as st
from PIL import Image, ImageOps
import cv2
import numpy as np
import random
import time
import seaborn as sns

from cv_funcs import *
from torchvision_funcs import *

@st.cache
def show_generated_image(image):
    st.image(image)
    
@st.cache(suppress_st_warning=True)
def randomize_palette_colors(n_rows, n_cols, palettes=['Set1', 'Set3', 'Spectral'], seed=time.time(), n_times=10):
    random.seed(seed)
    colors = [sns.color_palette(palette, n_rows*n_cols*n_times) for palette in palettes]
    _ = [random.shuffle(color) for color in colors]
    return colors

@st.cache(suppress_st_warning=True)
def remove_image_background(image):
    return deeplabv3_remove_bg(image)

title = 'Andy Warhol like Image Generator'
st.set_page_config(page_title=title, page_icon='favicon.jpeg', layout='centered')
st.title(title)
# iframeで埋め込みにして残りは非表示
st.components.v1.iframe(
    src="https://hf.space/streamlit/sentencebird/image-color-vectorization/", height=3000)
st.stop()

uploaded_file = st.file_uploader('Choose an image file')
if uploaded_file is None: uploaded_file = './sample.jpg'

if uploaded_file is not None:
    im = Image.open(uploaded_file)
    im.thumbnail((1000, 1000),resample=Image.BICUBIC) # resize
    
    is_masked = st.checkbox('With background masking? (3 colors)')
    if is_masked:
       im_masked, index_masked = remove_image_background(im)
       st.image(im_masked, caption='Masked image')        
    else: st.image(im, caption='Original')
    
    im_gray =  np.array(im.convert('L'))
    thresh, _img = cv2.threshold(im_gray, 0, 255, cv2.THRESH_OTSU)
    
    n_rows, n_cols = st.number_input('Rows', value=3), st.number_input('Columns', value=3)

    thresh = st.slider('Threshold', value=thresh, min_value=0.0, max_value=255.0)        
    colors = randomize_palette_colors(n_rows, n_cols, seed=0)
    
    if st.button('Shuffle colors'):
        colors = randomize_palette_colors(n_rows, n_cols, seed=time.time())            
    
    if True or st.button('Generate'):
        ims_generated = []

        for row in range(n_rows):
            for col in range(n_cols):
                i_color = n_cols * row + col
                rgbs = [np.array(color[i_color])*np.array([255, 255, 255]).tolist() for color in colors]
                ims_col = np.empty((*im_gray.shape, 3))
                for i in range(3): # RGB
                     ims_col[:, :, i] = (im_gray <= thresh) * rgbs[0][i] + (im_gray > thresh) * rgbs[1][i]
                     if is_masked: ims_col[:, :, i][index_masked] = rgbs[2][i]
                if col == 0:
                    im_col_concat = Image.fromarray(ims_col.astype(np.uint8))
                else:
                    im_col_concat = get_concat_h(im_col_concat, Image.fromarray(ims_col.astype(np.uint8)))
            if row == 0:
                im_generated = im_col_concat
            else:
                im_generated = get_concat_v(im_generated, im_col_concat)
#     if 'im_generated' in locals():
        st.image(im_generated)
