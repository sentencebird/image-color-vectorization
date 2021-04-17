import streamlit as st
from PIL import Image, ImageOps
import cv2
import numpy as np
import random
import time
import seaborn as sns

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def hsv_to_rgb(h, s, v):
    bgr = cv2.cvtColor(np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
    return [bgr[2]/255, bgr[1]/255, bgr[0]/255]

@st.cache
def show_generated_image(image):
    st.image(image)

@st.cache(suppress_st_warning=True)
def randomize_palette_colors(n_rows, n_cols, palette1="Set1", palette2="Set2", seed=0):
    random.seed(seed)
    colors1 = sns.color_palette(palette1, n_rows*n_cols)
    colors2 = sns.color_palette(palette2, n_rows*n_cols)    
    colors1, colors2 = random.sample(colors1, len(colors1)), random.sample(colors2, len(colors2))
    return colors1, colors2

@st.cache(suppress_st_warning=True)
def randomize_rgb_colors(n_rows, n_cols, seed=0):
    random.seed(seed)    
    colors1 = [[random.random() for j in range(3)] for i in range(n_rows*n_cols)] 
    colors2 = [[random.random() for j in range(3)] for i in range(n_rows*n_cols)]         
    return colors1, colors2

@st.cache(suppress_st_warning=True)
def randomize_hsv_colors(n_rows, n_cols, s=255, v=255, seed=0):
    random.seed(seed)    
    colors1 = [hsv_to_rgb(random.random()*180, s, v) for i in range(n_rows*n_cols)] 
    colors2 = [hsv_to_rgb(random.random()*180, s, v) for i in range(n_rows*n_cols)]         
    return colors1, colors2

title = 'Andy Warhol like Image Generator'
st.set_page_config(page_title=title, layout='centered')
st.title(title)
uploaded_file = st.file_uploader('Choose an image file')
if uploaded_file is None: uploaded_file = './sample.jpg'

if uploaded_file is not None:
    im = Image.open(uploaded_file)
    im.thumbnail((1000, 1000),resample=Image.BICUBIC) # resize    
    st.image(im, caption='Original')

    im_gray = np.array(im.convert('L'))
    thresh, _img = cv2.threshold(im_gray, 0, 255, cv2.THRESH_OTSU)
    
    n_rows, n_cols = st.number_input('Rows', value=3), st.number_input('Columns', value=3)
    
    # s = st.slider('Saturation', value=125.0, min_value=0.0, max_value=255.0)    
    # v = st.slider('Brightness', value=255.0, min_value=0.0, max_value=255.0)
    colors1, colors2 = randomize_palette_colors(n_rows, n_cols)
    thresh = st.slider('Threshold', value=thresh, min_value=0.0, max_value=255.0)
    
    if st.button('Shuffle colors'):
        colors1, colors2 = randomize_palette_colors(n_rows, n_cols, seed=time.time())

    if True or st.button('Generate'):
        im_bool = im_gray > thresh

        ims_generated = []
        for row in range(n_rows):
            for col in range(n_cols):
                i_color = n_cols * row + col
                rgb1, rgb2 = np.array(colors1[i_color])*np.array([255, 255, 255]).tolist(), np.array(colors2[i_color])*np.array([255, 255, 255]).tolist()
                ims_col = np.empty((*im_gray.shape, 3))
                for i in range(3): # RGB
                     ims_col[:, :, i] = (im_gray > thresh) * rgb1[i] + (im_gray <= thresh) * rgb2[i]
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
