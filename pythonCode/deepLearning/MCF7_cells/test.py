from skimage import io, transform
import matplotlib.pyplot as plt
img_name = 'data/cells/imgTotal/imageAfter_0.tif'
img_name1 = 'data/faces/10comm-decarlo.jpg'
image = io.imread(img_name1)
plt.imshow(image)
plt.show()
print(image.shape)
