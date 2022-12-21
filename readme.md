# Introduction

this project contains all the codes and data used for the paper: "Flame-state monitoring based on very low number of visible or infrared images via few-shot learning", a paper talks about using few shot learning algorithms to realize combustion monitoring, preprint is here:

[[2210.07845] Realizing Flame State Monitoring with Very Few Visual or Infrared Images via Few-Shot Learning](https://arxiv.org/abs/2210.07845) 

## Structure of the project

the project contains two folders: Prototypical_Network, Siamese_kNN. In both folder, contains the train, test and feature visualization codes for these two algorithms.

# About Data availability

the dataset has been uploaded to Kaggle, please refer to:

```
@misc{ruiyuan kang_2022,
 title={flame state classification},
 url={https://www.kaggle.com/ds/2750725},
 DOI={10.34740/KAGGLE/DS/2750725},
 publisher={Kaggle},
 author={Ruiyuan Kang},
 year={2022}
}
```

Which includes the images used for training, validation and test, also the features used for visualization, just put them respectively into the folder of the algorithm, more hints can be found from the code hyperparameter cofigurations.
