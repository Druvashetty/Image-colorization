import numpy as np
import cv2
import matplotlib.pyplot as plt

print("loading models.....")
net = cv2.dnn.readNetFromCaffe('./model/colorization_deploy_v2.prototxt','./model/colorization_release_v2.caffemodel')
pts = np.load('./model/pts_in_hull.npy')

class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2,313,1,1)

net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1,313],2.606,dtype='float32')]

image = cv2.imread('./image3/img2.jpg')  # Replace with your image path
scaled = image.astype("float32")/255.0
lab = cv2.cvtColor(scaled,cv2.COLOR_BGR2LAB)

resized = cv2.resize(lab,(224,224))
L = cv2.split(resized)[0]
L -= 50

net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1,2,0))

ab = cv2.resize(ab, (image.shape[1],image.shape[0]))

L = cv2.split(lab)[0]
colorized = np.concatenate((L[:,:,np.newaxis], ab), axis=2)

colorized = cv2.cvtColor(colorized,cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized,0,1)

colorized = (255 * colorized).astype("uint8")

cv2.imshow("Original",image)
cv2.imshow("Colorized",colorized)

cv2.waitKey(0)
cv2.destroyAllWindows() # Close the OpenCV windows

# --- Plotting Graphs ---

# 1. Accuracy vs Epochs (Hypothetical Example)
epochs = np.arange(1, 11)  # Example epochs
accuracy = [0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98]  # Example accuracy values

plt.figure(figsize=(8, 6))
plt.plot(epochs, accuracy, marker='o', linestyle='-', color='b')
plt.title("Accuracy vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

# 2. Loss vs Epochs Pie Chart (Hypothetical Example)
loss = [0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.18, 0.15, 0.12, 0.1]  # Example loss values

plt.figure(figsize=(8, 6))
plt.pie(loss, labels=None, autopct='%1.1f%%', startangle=90)
plt.title("Loss Distribution Across Epochs")
plt.axis('equal')
plt.legend(loss, title="Loss Values", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.show()

bgr_colorized = cv2.split(colorized)

# 3 Flatten the channels
b_channel = bgr_colorized[0].flatten()
g_channel = bgr_colorized[1].flatten()
r_channel = bgr_colorized[2].flatten()

plt.figure(figsize=(8, 6))
plt.bar( ['Blue', 'Green', 'Red'], [b_channel.mean(), g_channel.mean(), r_channel.mean()])
plt.title("Average Color Distribution in Colorized Image")
plt.xlabel("Channels")
plt.ylabel("Average Pixel Intensity")
plt.grid(True)
plt.show()

