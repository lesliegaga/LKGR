from src.train import *
import argparse, sys
from os.path import join
from utility.data_loader import data_split

PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PATH)
ROOT = join(PATH, '../')
sys.path.append(ROOT)

if __name__ == '__main__':
    #   book
    parser = argparse.ArgumentParser(description='Parse for LKGR.')
    parser.add_argument('--data_dir', type=str, default=f'{PATH}/data/', help='file path of datasets.')
    parser.add_argument('--data_name', type=str, default='book', help='select a dataset, e.g., last-fm.')
    parser.add_argument('--kg_file', type=str, default='kg_final.txt', help='select kg file.')
    parser.add_argument('--gpu_id', type=int, default=0, help='select gpu_id')
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--node_dim', type=int, default=64, help='the dimension of users, items and entities')
    parser.add_argument('--n_layer', type=int, default=1, help='the number of layers')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size.')
    parser.add_argument('--sample_size', type=int, default=8, help='the size of neighbors sampled')
    parser.add_argument('--agg_type', type=str, default='sum', help='specify the type of aggregation for users from [sum, concat, ngh]')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout probability w.r.t. message dropout for each deep layer. 0: no dropout.')
    parser.add_argument('--l2_weight', type=float, default=5e-7, help='Lambda when calculating CF l2 loss.')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epoch.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--curvature', type=float, default=1., help='the curvature of the manifold.')

    args = parser.parse_args()
    saved_dir = 'logs/{}/Dim{}/'.format(args.data_name, args.node_dim)
    args.saved_dir = saved_dir

    # data_split(args)

    Exp_run(args)