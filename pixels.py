#source : freeCodeCamp

#get image and convert to pixels
from PIL import Image
import numpy as np
img = Image.open('five1.png')
data = list(img.getdata())

for i in range(len(data)):
    data[i] = 255 - data[i]
print(data)

five = np.array(data)/256

print(five)
