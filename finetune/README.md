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
