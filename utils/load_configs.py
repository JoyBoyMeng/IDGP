import argparse
import sys
import torch

def get_popularity_prediction_args():
    """
    get the args for the popularity prediction task
    :return:
    """
    # arguments
    parser = argparse.ArgumentParser('Interface for the popularity prediction task')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='aminer', choices=['aminer', 'aps', 'yelp', 'twitter'])
    parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--model_name', type=str, default='DyGFormer', help='name of the model',
                        choices=['JODIE', 'DyRep', 'TGAT', 'TGN', 'TGN-id', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer', 'Fluxion'])
    parser.add_argument('--gpu', type=int, default=0, help='number of gpu to use')
    parser.add_argument('--num_neighbors', type=int, default=20, help='number of neighbors to sample for each node')
    parser.add_argument('--sample_neighbor_strategy', type=str, default='recent', choices=['uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
    parser.add_argument('--time_scaling_factor', default=1e-6, type=float, help='the hyperparameter that controls the sampling preference with time interval, '
                        'a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, '
                        'it works when sample_neighbor_strategy == time_interval_aware')
    parser.add_argument('--num_walk_heads', type=int, default=8, help='number of heads used for the attention in walk encoder')
    parser.add_argument('--num_heads', type=int, default=2, help='number of heads used in attention layer')
    parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
    parser.add_argument('--walk_length', type=int, default=1, help='length of each random walk')
    parser.add_argument('--time_gap', type=int, default=2000, help='time gap for neighbors to compute node features')
    parser.add_argument('--time_feat_dim', type=int, default=100, help='dimension of the time embedding')
    parser.add_argument('--position_feat_dim', type=int, default=172, help='dimension of the position embedding')
    parser.add_argument('--patch_size', type=int, default=1, help='patch size')
    parser.add_argument('--channel_embedding_dim', type=int, default=50, help='dimension of each channel embedding')
    parser.add_argument('--max_input_sequence_length', type=int, default=32, help='maximal length of the input sequence of each node')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'], help='name of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stopping')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='ratio of test set')
    parser.add_argument('--num_runs', type=int, default=5, help='number of runs')
    parser.add_argument('--test_interval_epochs', type=int, default=10, help='how many epochs to perform testing once')
    parser.add_argument('--load_best_configs', action='store_true', default=False, help='whether to load the best configurations')
    parser.add_argument('--dataset_time_gap', type=str, default='half_year', help='time interval for popularity prediction')
    parser.add_argument('--router_mode', type=str, default='linear', help='router for fluxion decision')
    parser.add_argument('--fluxion_select_num', type=int, default=3, help='fluxion numbers selected in fluxion router')
    parser.add_argument('--t_start', type=int, default=2, help='router temperature start')
    parser.add_argument('--t_end', type=int, default=0.5, help='router temperature end')
    parser.add_argument('--t_steps', type=int, default=5000, help='router temperature change step num')
    parser.add_argument('--exp', type=str, default='debug', help='just distinguish which exp it is')
    parser.add_argument('--fluxion_member_num', type=int, default=16, help='node num in one fluxion')
    parser.add_argument('--fluxion_init_type', type=str, default='zero', help='fluxion init with zero or ball')
    parser.add_argument('--distance_type', type=str, default='cosine', help='fluxion router metric euclidean or cosine')
    parser.add_argument('--ema', type=float, default=0.3, help='ratio of ema_momentum')

    try:
        args = parser.parse_args()
        args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    except:
        parser.print_help()
        sys.exit()

    assert args.dataset_name in ['aminer', 'aps', 'yelp', 'twitter'], f'Wrong value for dataset_name {args.dataset_name}!'
    if args.load_best_configs:
        load_node_classification_best_configs(args=args)
    return args
