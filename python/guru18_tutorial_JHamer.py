import matplotlib.pyplot as plt
plt.figure(figsize=(12,18))
img = plt.imread("./1_prep_guru18.png")
plt.imshow(img)
plt.title("SSH and prep guru18", fontsize=30)
plt.show()

plt.figure(figsize=(26,20))
img = plt.imread("./launch_background_pod_with_jupyter.png")
plt.imshow(img)
plt.title("Launch a background pod - which will stay active after exiting shell - WITH Jupyter", fontsize=30)
plt.show()

