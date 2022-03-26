import torch
from dataset import TeethPairDataset
import numpy as np


def default_collate_pair_fn(list_data):
    feats0, feats1, matching_inds, trans = list(zip(*list_data))
    feats_batch0 = []
    feats_batch1 = []
    matching_inds_batch, trans_batch, len_batch = [], [], []
    batch_id = 0
    for batch_id, _ in enumerate(feats0):

        N0 = feats0[batch_id].shape[0]
        N1 = feats1[batch_id].shape[0]

        # Move batchids to the beginning
        feats_batch0.append(torch.from_numpy(feats0[batch_id]))
        feats_batch1.append(torch.from_numpy(feats1[batch_id]))

        trans_batch.append(torch.from_numpy(trans[batch_id]))

        if len(matching_inds[batch_id]) == 0:
            matching_inds[batch_id].extend([0, 0])

        matching_inds_batch.append(
            torch.from_numpy(np.array(matching_inds[batch_id])))
        len_batch.append([N0, N1])

    # Concatenate all lists
    feats_batch0 = torch.stack(feats_batch0, 0).float()
    feats_batch1 = torch.stack(feats_batch1, 0).float()
    trans_batch = torch.stack(trans_batch, 0).float()

    return {
        'sinput0_F': feats_batch0,
        'sinput1_F': feats_batch1,
        'correspondences': matching_inds_batch,
        'trans': trans_batch,
        'len_batch': len_batch,
    }


def make_data_loader(args):
    dataset = TeethPairDataset(phase = 'train', args = args)
    collate_pair_fn = default_collate_pair_fn
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_pair_fn,
        pin_memory=False)
    return loader
