import cv2
from PIL import Image
import numpy as np

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

# def remove_bg(
#     path,
#     BLUR = 21,
#     CANNY_THRESH_1 = 10,
#     CANNY_THRESH_2 = 200,
#     MASK_DILATE_ITER = 10,
#     MASK_ERODE_ITER = 10,
#     MASK_COLOR = (0.0,0.0,1.0),
# ):
#     img = cv2.imread(path)
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#     edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
#     edges = cv2.dilate(edges, None)
#     edges = cv2.erode(edges, None)

#     contour_info = []
#     contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
#     for c in contours:
#         contour_info.append((
#             c,
#             cv2.isContourConvex(c),
#             cv2.contourArea(c),
#         ))
#     contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
#     max_contour = contour_info[0]

#     mask = np.zeros(edges.shape)
#     cv2.fillConvexPoly(mask, max_contour[0], (255))

#     mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
#     mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
#     mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
#     mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask

#     mask_stack  = mask_stack.astype('float32') / 255.0          # Use float matrices, 
#     img         = img.astype('float32') / 255.0                 #  for easy blending

#     masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR) # Blend
#     masked = (masked * 255).astype('uint8')                     # Convert back to 8-bit 

#     c_blue, c_green, c_red = cv2.split(img)

#     img_a = cv2.merge((c_red, c_green, c_blue, mask.astype('float32') / 255.0))
#     index = np.where(img_a[:, :, 3] == 0)
#     #img_a[index] = [1.0, 1.0, 1.0, 1.0]
#     return img_a