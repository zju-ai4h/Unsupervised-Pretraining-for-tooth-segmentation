import numpy as np
import trimesh
from plyfile import PlyData,PlyElement
mean_centroids = np.array([[1.60083597, -3.74032063, 2.33078412],
                                [0.5687525, -6.21317224, -6.30563364]])

def get_triangle_struct(mesh):
    return (mesh.triangles - np.expand_dims(mesh.triangles_center, axis=1)).reshape(-1, 9)

def readPredFile(labelfile):
    label = []
    with open(labelfile, 'r') as file:
        lines = file.readlines()
        for s, line in enumerate(lines):
            label.append(int(line))
    return np.array(label)
  
for i in range(len(ply_fold)):
    ply_name = ply_fold[i].replace('', '')
    stl_name = ply_fold[i].replace('.ply', '.stl')

    mesh = trimesh.load(stl_name)
    mesh.fix_normals()
    if stl_name.find('l_aligned.stl') != -1:
        category = 0
    elif stl_name.find('u_aligned.stl') != -1:
        category = 1
    
    txt_name = stl_name.replace('_aligned.stl', '.txt')
    l = readPredFile(txt_name)
    mesh.apply_translation(-mean_centroids[category])
    xyz = np.array(mesh.triangles_center)
    inf_pts = np.linspace(0, xyz.shape[0] - 1, 20000).astype(int).tolist()
    # sampling to 20,000
    xyz = xyz[inf_pts, :]
    normals = np.array(mesh.face_normals)[inf_pts, :]
    structs = get_triangle_struct(mesh)[inf_pts, :]
    features = np.hstack((structs, normals))
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    id = [0, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]
    labels = np.ones(l.shape[0])
    for i in range(len(id)):
        labels[np.where(l == id[i])] = label[i]
    labels = labels[inf_pts]


    point_cloud = np.array(
        [(x[i], y[i], z[i], features[i][0],features[i][1],features[i][2],features[i][3],features[i][4],
          features[i][5],features[i][6],features[i][7],features[i][8],features[i][9],features[i][10],features[i][11], features[i][12], labels[i]) for i in range(x.shape[0])],
        dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),('f1', 'f4'),('f2', 'f4'),('f3', 'f4'),('f4', 'f4'),('f5', 'f4'),('f6', 'f4'),
               ('f7', 'f4'),('f8', 'f4'),('f9', 'f4'),('f10', 'f4'),('f11', 'f4'),('f12', 'f4'),('label', 'u1')]
    )
    el = PlyElement.describe(point_cloud, 'vertex', comments=['vertices'])
    PlyData([el]).write(ply_name)
