import logging
import torch.utils.data
import numpy as np
import os
import copy
from scipy.linalg import expm, norm
import trimesh
import MinkowskiEngine as ME
import open3d as o3d


def make_open3d_point_cloud(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


def get_matching_indices(source, target, trans, search_voxel_size, K=None):
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    source_copy.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(target_copy)

    match_inds = []
    for i, point in enumerate(source_copy.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            match_inds.append((i, j))
    return match_inds


def get_triangle_struct(mesh):
    return (mesh.triangles - np.expand_dims(mesh.triangles_center, axis=1)).reshape(-1, 9)


# Rotation matrix along axis with angle theta
def M(axis, theta):
    return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


def sample_random_trans(pcd, randg, rotation_range=360):
    T = np.eye(4)
    R = M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
    T[:3, :3] = R
    T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
    return T


def select_area(features0, features1, select_min = 0.4, select_max = 0.7):
    rd = np.random.uniform(select_min, select_max)
    N = int(np.floor(features0.shape[0] * rd))
    start = np.random.randint(0, features0.shape[0] - N)
    v1_area = features0[0:(start + N), :]
    v2_area = features1[start:, :]
    return v1_area, v2_area


def choose_point(sel):
    inf_pts = np.linspace(0, len(sel) - 1, 10000).astype(int).tolist()
    return inf_pts
    

class PairDataset(torch.utils.data.Dataset):

    def __init__(self,
                 phase,
                 args,
                 manual_seed=False):
        self.phase = phase
        self.files = []
        self.data_objects = []
        self.voxel_size = args.voxel_size
        self.matching_search_voxel_size = \
            args.voxel_size * args.positive_pair_search_voxel_size_multiplier
        self.config = args
        self.randg = np.random.RandomState()
        self.mean_centroids = np.array([[1.60083597, -3.74032063, 2.33078412],
                                        [0.5687525, -6.21317224, -6.30563364]])
        self.root = '/'
        
        
        if manual_seed:
            self.reset_seed()

        if phase == "train":
            self.root_filelist = root = args.teeth_dir
        else:
            raise NotImplementedError

        logging.info(f"Loading the subset {phase} from {root}")
        fname_txt = os.path.join(self.root_filelist)
        with open(fname_txt) as f:
            content = f.readlines()
        fnames = [x.strip().split() for x in content]
        for fname in fnames:
            self.files.append(fname)
        logging.info(f'[DATASET] {len(self.files)} instances were loaded')

    def reset_seed(self, seed=0):
        logging.info(f"Resetting the data loader seed to {seed}")
        self.randg.seed(seed)

    def __len__(self):
        return len(self.files)


class TeethPairDataset(PairDataset):

    def __init__(self,
                 phase,
                 args,
                 manual_seed=False):
        
        PairDataset.__init__(self, phase, args, manual_seed)

    def get_correspondences(self, idx):
        file = os.path.join(self.files[idx][0])
        mesh = trimesh.load_mesh(file)
        mesh.fix_normals()
        if file.find('l_aligned.stl') != -1:
            category = 0
        elif file.find('u_aligned.stl') != -1:
            category = 1
            
        mesh.apply_translation(-self.mean_centroids[category])
         
        pcd = mesh.triangles_center
        matching_search_voxel_size = self.matching_search_voxel_size

        T0 = sample_random_trans(pcd, self.randg)
        T1 = sample_random_trans(pcd, self.randg)
        trans = T1 @ np.linalg.inv(T0)

        mesh_copy = copy.deepcopy(mesh)
        mesh0 = mesh.apply_transform(T0)
        mesh1 = mesh_copy.apply_transform(T1)

        features0 = np.hstack((mesh0.triangles_center, get_triangle_struct(mesh0), mesh0.face_normals))
        features1 = np.hstack((mesh1.triangles_center, get_triangle_struct(mesh1), mesh1.face_normals))
        view1, view2 = select_area(features0, features1)

        xyz0 = view1[:, 0:3]
        xyz1 = view2[:, 0:3]

        # Voxelization
        sel0 = ME.utils.sparse_quantize(xyz0 / self.voxel_size, return_index=True)
        sel1 = ME.utils.sparse_quantize(xyz1 / self.voxel_size, return_index=True)
        sel0 = sel0[1]
        sel1 = sel1[1]

        inf_pts0 = choose_point(sel0)
        inf_pts1 = choose_point(sel1)

        sel0 = sel0[inf_pts0]
        sel1 = sel1[inf_pts1]

        pcd0 = make_open3d_point_cloud(xyz0)
        pcd1 = make_open3d_point_cloud(xyz1)

        pcd0.points = o3d.utility.Vector3dVector(np.array(pcd0.points)[sel0])
        pcd1.points = o3d.utility.Vector3dVector(np.array(pcd1.points)[sel1])

        matches = get_matching_indices(pcd0, pcd1, trans, matching_search_voxel_size)

        feats0 = view1[sel0]
        feats1 = view2[sel1]

        return (feats0, feats1, matches, trans)

    def __getitem__(self, idx):
        return self.get_correspondences(idx)
