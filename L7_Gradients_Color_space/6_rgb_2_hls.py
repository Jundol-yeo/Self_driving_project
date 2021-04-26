import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# Read an image
image = mpimg.imread('test4.jpg')

def hls_select(img, thresh=(0, 255)):

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    schannel = hls[:, :, 2]
    binary_output = np.zeros_like(schannel)
    binary_output[(schannel > thresh[0]) & (schannel <= thresh[1])] = 1

    print(hls)
    return binary_output

# Optional TODO - tune the threshold to try to match the above image!    
hls_binary = hls_select(image, thresh=(90, 255))

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(hls_binary, cmap='gray')
ax2.set_title('Thresholded S', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
