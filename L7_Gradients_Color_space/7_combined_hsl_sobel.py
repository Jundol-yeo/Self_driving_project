import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

#Read an image
image = mpimg.imread('test1.jpg')

#convert the image to hls and gray
image1 = np.copy(image)
hls = cv2.cvtColor(image1, cv2.COLOR_RGB2HLS)
#gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def pipeline(img, s_thresh=(100, 255), sobelx_thresh = (20, 100)):

    s_channel = hls[:, :, 2]

    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    sx_binary = np.zeros_like(scaled_sobel)
    sx_binary[(scaled_sobel >= sobelx_thresh[0]) & (scaled_sobel <= sobelx_thresh[1])] = 1

    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel >= s_thresh[1])] = 1

    color_binary = np.dstack((np.zeros_like(sx_binary), sx_binary, s_binary)) * 255
    
    combined_binary = np.zeros_like(sx_binary)
    combined_binary[(s_binary == 1) | (sx_binary == 1)] = 1
    
    return color_binary, combined_binary
    
result1, result2 = pipeline(image)

# Plot the result
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
f.tight_layout()

ax1.imshow(image)
ax1.set_title('Original Image', fontsize=40)

ax2.imshow(result1)
ax2.set_title('Pipeline Result', fontsize=40)

ax3.imshow(result2, cmap = 'gray')
ax3.set_title('Pipeline Result', fontsize=40)

plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()