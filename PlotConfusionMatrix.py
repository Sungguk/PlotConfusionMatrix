
# coding: utf-8

# In[1]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


# In[2]:


num_of_classes=320
random_matrix = np.random.random((num_of_classes,num_of_classes))
classes = [str(i) for i in range(num_of_classes)]


# In[3]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    matplotlib.rcParams.update({'font.size': 5})
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.1f' if normalize else '.1f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(50,50)
    fig.savefig('test_confusion_matrix.png',dpi=150)
    


# In[ ]:


plot_confusion_matrix(random_matrix, classes=classes,
                      title='Confusion matrix, without normalization')

