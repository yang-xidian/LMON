import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")
device = torch.device('cpu')
texas = torch.load("wisconsin_file")
print(texas.shape)
texas_labels = torch.load("wisconsin_labels")
print(texas_labels.shape)
texas_labels=texas_labels.argmax(1)

tsne = TSNE(n_components=2, init='pca',random_state=501)

texas=texas.cpu().numpy()
texas_labels=texas_labels.cpu().numpy()
x_tsne = tsne.fit_transform(texas)

x_min, x_max = x_tsne.min(0), x_tsne.max(0)
x_norm = (x_tsne-x_min) / (x_max-x_min)

plt.figure(figsize=(8,8))
for i in range(x_norm.shape[0]):
    plt.text(x_norm[i,0],x_norm[i,1],'·',size=30,color=plt.cm.Set1(texas_labels[i]))


plt.xticks([])
plt.yticks([])
plt.title(label="Visualization of wisconsin data after embedding",
          fontdict={
              "fontsize":14,
          }
          )
#plt.show()
plt.savefig("wisconsin.png", dpi=700)