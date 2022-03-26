from torch.utils.data import Dataset
from plyfile import PlyData
import numpy as np
import os

def read_txt(path):
  """Read txt file into lines.
  """
  with open(path) as f:
    lines = f.readlines()
  lines = [x.strip() for x in lines]
  return lines

def generate_findices(face_numbers, num_points):
    final_list = []
    inf_pts = np.linspace(0, face_numbers - 1, num_points).astype(int).tolist()
    max_batch = 1
    for i in range(max_batch):
        final_list += inf_pts[i::max_batch]
    final_list += list(set(range(face_numbers)) - set(inf_pts))
    return np.array(final_list)


class TeethDataset(Dataset):

    def __init__(self, num_points=20000, partition='train'):
        self.num_points = num_points
        self.partition = partition
        self.num_categories = 2
        self.DATA_PATH_FILE = {
              'train': 'teeth_train.txt',
              'val': 'teeth_val.txt',
              'test': 'teeth_test.txt'
        }
        self.data_paths = read_txt(os.path.join('./dataset', self.DATA_PATH_FILE[self.partition]))

    def load_ply(self, index):
        filepath = self.data_paths[index]
        plydata = PlyData.read(filepath)
        data = plydata.elements[0].data
        feats = np.array([data['x'], data['y'], data['z'],data['f1'], data['f2'], data['f3'],data['f4'],data['f5'],data['f6'],data['f7'],
                          data['f8'],data['f9'],data['f10'],data['f11'],data['f12']], dtype=np.float32).T
        label = np.array(data['label'], dtype=np.int32)
        if filepath.find('l_aligned')!=-1:
            category = 0
        if filepath.find('u_aligned')!=-1:
            category = 1
        return feats, label, category
    
    
    def __getitem__(self, index):
        feats, label, category = self.load_ply(index)
        one_hot_categories = np.zeros(self.num_categories)
        one_hot_categories[category] = 1
        return feats,label,one_hot_categories
        
    
    def __len__(self):
        num_data = len(self.data_paths)
        return num_data   
