For fine-tuning, our codebase is based on [AnTao97/*dgcnn*.pytorch](https://github.com/AnTao97/dgcnn.pytorch).  
Here we introduce the format before and after the preprocessing. Original dataset is ``.txt + .stl`` format, ``.txt`` contains the label of every surface and  ``.stl`` is a 3-Dimensional Intraoral Mesh Scan. For better management, we extract the features of ``.stl``, sample 20,000 surfaces and add labels to it. 
# Original dataset  
trainset    
├── 00000     
│   ├── 00000.stl  
│   ├── 00000.txt  
│    ……  
├── XXXXX      
│   ├── XXXXX.stl  
│   └── XXXXX.txt  
# Preprocess dataset  
trainset     
├── 00000.ply    
 ……     
└── XXXXX.ply    
``data_preprocess/process.py`` explains how to extract features and convert to ``.ply`` format， but it can't run directly. You can refer to our description below and the comments in the code to preprocess your data. First, you need to have a list that stores all the raw data, e.g. [[``00000.stl``,``00000.txt``],...,[``XXXXX.stl``,``XXXXX.txt``]]. Then, you can use the provided code to extract the feature of each face and downsample. Finally, you need to map the labels of teeth to 0-33.
