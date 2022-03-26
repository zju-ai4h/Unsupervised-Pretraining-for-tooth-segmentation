import argparse

# default parameters
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type = bool, default = True)

    parser.add_argument('--teeth_dir', type = str, default = './dataset/overlap.txt')

    parser.add_argument('--voxel_size', type = float, default = 0.3)

    parser.add_argument('--positive_pair_search_voxel_size_multiplier', type = float, default = 1.5)

    parser.add_argument('--k', type = int, default = 25)

    parser.add_argument('--seed', type = int, default = 123)

    parser.add_argument('--lr', type = float, default = 0.001)

    parser.add_argument('--exp_gamma', type = float, default = 0.99)

    parser.add_argument('--weight_decay', type = float, default = 1e-4)

    parser.add_argument('--epochs', type = int, default = 20)

    # continue training
    parser.add_argument('--weight', type = str, default = None)

    parser.add_argument('--lenient_weight_loading', type = bool, default = False)

    parser.add_argument('--nceT', type = float, default = 0.4)

    parser.add_argument('--npos', type = int, default = 4096)

    parser.add_argument('--lr_update_freq', type = int, default = 1000)

    parser.add_argument('--exp_name', type = str, default = 'pretrain')

    parser.add_argument('--batch_size', type = int, default = 4)
    args = parser.parse_args()
    return args
