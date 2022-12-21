import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

#%% load data
test_feature=np.loadtxt('data_save/test_fea_2.txt')
support_feature =np.loadtxt('data_save/support_fea.txt')
orig_label=np.loadtxt('data_save/label_list.txt')
pred_label=np.loadtxt('data_save/pred_list.txt')

#%%
all_feature_set=np.concatenate((support_feature,test_feature),axis=0)
support_label=np.arange(0,6)

#%% TSNE
tsne=TSNE(init='pca',random_state=555)
embedding=tsne.fit_transform(all_feature_set)
x_min, x_max = embedding.min(0), embedding.max(0)
all_norm = (embedding - x_min) / (x_max - x_min)
support_norm=all_norm[0:6]
X_norm=all_norm[6:]

#%%
data_dic={}
label_collect={}
plt.figure(figsize=[6,3])
for label in range(6):
    ele_index=np.where(pred_label==label)
    data_dic[int(label)]=X_norm[ele_index]
    plt.plot(data_dic[int(label)][:,0], data_dic[int(label)][:,1],  '^', label='Class ' + str(label + 1),markersize=3)
plt.plot(support_norm[:,0],support_norm[:,1], 'oy',label='Prototypes',markersize=7.,markeredgecolor='b')
diff_index=np.where(pred_label !=orig_label)
diff_data=X_norm[diff_index]
plt.plot(diff_data[:,0],diff_data[:,1],'xk',label='Misclassified',markersize=4.)
plt.legend(prop={'family' : 'Times New Roman', 'size'   : 10},loc='upper left', ncol=2,markerscale=1.3,fancybox=False)
plt.yticks([])
plt.xticks([])

plt.tight_layout()
plt.show()


