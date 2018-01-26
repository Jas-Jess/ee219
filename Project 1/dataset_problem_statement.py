import numpy as np
from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt; plt.rcdefaults()

# Load the training data from Computer Technology
# subset = 'train' means that we are loading the training data (we input test if we want the testing data)
# categories takes in the dataset name you want
# random_state is numpy random number generator
graphics_train = fetch_20newsgroups(subset='train', categories=['comp.graphics'], shuffle=True, random_state=42)
misc_train = fetch_20newsgroups(subset='train', categories=['comp.os.ms-windows.misc'], shuffle=True, random_state=42)
pchardware_train = fetch_20newsgroups(subset='train', categories=['comp.sys.ibm.pc.hardware'], shuffle=True, random_state=42)
machardware_train = fetch_20newsgroups(subset='train', categories=['comp.sys.mac.hardware'], shuffle=True, random_state=42)

# Load the training data from the recreational activity
auto_train = fetch_20newsgroups(subset='train', categories=['rec.autos'], shuffle=True, random_state=42)
motorcycles_train = fetch_20newsgroups(subset='train', categories=['rec.motorcycles'], shuffle=True, random_state=42)
baseball_train = fetch_20newsgroups(subset='train', categories=['rec.sport.baseball'], shuffle=True, random_state=42)
hockey_train = fetch_20newsgroups(subset='train', categories=['rec.sport.hockey'], shuffle=True, random_state=42)

# Plotting the histogram
objects = ('graphics', 'ms-windows.misc', 'pc.hardware', 'mac.hardware', 'autos', 'motorcycles', 'baseball', 'hockey')
y_pos = np.arange(len(objects))
data_num = [len(graphics_train.data), len(misc_train.data), len(pchardware_train.data), len(machardware_train.data), len(auto_train.data), len(motorcycles_train.data), len(baseball_train.data), len(hockey_train.data)]

plt.bar(y_pos, data_num, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('# of Training Documents')
plt.title('Subclasses of \'Computer Technology\' & \'Recreational Activity\'')

plt.show()