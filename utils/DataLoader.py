from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pandas as pd


class CustomizedDataset(Dataset):
    def __init__(self, indices_list: list):
        """
        Customized dataset.
        :param indices_list: list, list of indices
        """
        super(CustomizedDataset, self).__init__()

        self.indices_list = indices_list

    def __getitem__(self, idx: int):
        """
        get item at the index in self.indices_list
        :param idx: int, the index
        :return:
        """
        return self.indices_list[idx]

    def __len__(self):
        return len(self.indices_list)


def get_idx_data_loader(indices_list: list, batch_size: int, shuffle: bool):
    """
    get data loader that iterates over indices
    :param indices_list: list, list of indices
    :param batch_size: int, batch size
    :param shuffle: boolean, whether to shuffle the data
    :return: data_loader, DataLoader
    """
    dataset = CustomizedDataset(indices_list=indices_list)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=False)
    return data_loader


class Data:

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray,
                 edge_ids: np.ndarray, labels: np.ndarray, slow_features: np.ndarray, sp_features: np.ndarray,
                 timeinterval_features: np.ndarray):
        """
        Data object to store the nodes interaction information.
        :param src_node_ids: ndarray
        :param dst_node_ids: ndarray
        :param node_interact_times: ndarray
        :param edge_ids: ndarray
        :param labels: ndarray
        """
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)
        self.num_unique_item_nodes = len(set(dst_node_ids))
        self.num_unique_user_nodes = len(set(src_node_ids))
        self.slow_features = slow_features
        self.sp_features = sp_features
        self.timeinterval_features = timeinterval_features


def get_popularity_prediction_data(dataset_name: str, val_ratio: float, test_ratio: float, time_interval: str):
    """
    generate data for node classification task
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, (Data object)
    """
    # Load data and train val test split
    if dataset_name in ['aminer', 'yelp', 'aps', 'twitter']:
        train_data = pd.read_csv('./processed_data/{}/pp_{}_half_year_train.csv'.format(dataset_name, dataset_name))
        val_data = pd.read_csv('./processed_data/{}/pp_{}_half_year_val.csv'.format(dataset_name, dataset_name))
        test_data = pd.read_csv('./processed_data/{}/pp_{}_half_year_test.csv'.format(dataset_name, dataset_name))
        all_data = pd.concat([train_data, val_data, test_data])
        train_data = train_data.to_numpy()
        val_data = val_data.to_numpy()
        test_data = test_data.to_numpy()
        all_data = all_data.to_numpy()

        full_slow_feature = np.load(
            './processed_data/{}/pp_{}_half_year_all_feature.npy'.format(dataset_name, dataset_name))
        train_slow_feature = np.load(
            './processed_data/{}/pp_{}_half_year_train_feature.npy'.format(dataset_name, dataset_name))
        val_slow_feature = np.load(
            './processed_data/{}/pp_{}_half_year_val_feature.npy'.format(dataset_name, dataset_name))
        test_slow_feature = np.load(
            './processed_data/{}/pp_{}_half_year_test_feature.npy'.format(dataset_name, dataset_name))

        full_sp_feature = np.load(
            './processed_data/{}/pp_{}_half_year_all_sp.npy'.format(dataset_name, dataset_name))
        train_sp_feature = np.load(
            './processed_data/{}/pp_{}_half_year_train_sp.npy'.format(dataset_name, dataset_name))
        val_sp_feature = np.load(
            './processed_data/{}/pp_{}_half_year_val_sp.npy'.format(dataset_name, dataset_name))
        test_sp_feature = np.load(
            './processed_data/{}/pp_{}_half_year_test_sp.npy'.format(dataset_name, dataset_name))

        full_time_feature = np.load(
            './processed_data/{}/pp_{}_half_year_all_dt_N200.npy'.format(dataset_name, dataset_name))
        train_time_feature = np.load(
            './processed_data/{}/pp_{}_half_year_train_dt_N200.npy'.format(dataset_name, dataset_name))
        val_time_feature = np.load(
            './processed_data/{}/pp_{}_half_year_val_dt_N200.npy'.format(dataset_name, dataset_name))
        test_time_feature = np.load(
            './processed_data/{}/pp_{}_half_year_test_dt_N200.npy'.format(dataset_name, dataset_name))

        full_data = Data(src_node_ids=all_data[:, 0].astype('int64'), dst_node_ids=all_data[:, 1].astype('int64'),
                         node_interact_times=all_data[:, 2].astype('int64'),
                         edge_ids=all_data[:, 4].astype('int64'), labels=all_data[:, 3].astype('float'),
                         slow_features=full_slow_feature.astype('int64'),
                         sp_features=full_sp_feature, timeinterval_features=full_time_feature)
        train_data = Data(src_node_ids=train_data[:, 0].astype('int64'), dst_node_ids=train_data[:, 1].astype('int64'),
                          node_interact_times=train_data[:, 2].astype('int64'),
                          edge_ids=train_data[:, 4].astype('int64'), labels=train_data[:, 3].astype('float'),
                          slow_features=train_slow_feature.astype('int64'),
                          sp_features=train_sp_feature, timeinterval_features=train_time_feature)
        val_data = Data(src_node_ids=val_data[:, 0].astype('int64'), dst_node_ids=val_data[:, 1].astype('int64'),
                        node_interact_times=val_data[:, 2].astype('int64'),
                        edge_ids=val_data[:, 4].astype('int64'), labels=val_data[:, 3].astype('float'),
                        slow_features=val_slow_feature.astype('int64'),
                        sp_features=val_sp_feature, timeinterval_features=val_time_feature)
        test_data = Data(src_node_ids=test_data[:, 0].astype('int64'), dst_node_ids=test_data[:, 1].astype('int64'),
                         node_interact_times=test_data[:, 2].astype('int64'),
                         edge_ids=test_data[:, 4].astype('int64'), labels=test_data[:, 3].astype('float'),
                         slow_features=test_slow_feature.astype('int64'),
                         sp_features=test_sp_feature, timeinterval_features=test_time_feature)
        # the setting of seed follows previous works
        random.seed(2025)
        NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
        unique_node_ids_num = full_data.num_unique_nodes + 1
        edge_ids_num = full_data.num_interactions + 1
        node_raw_features = np.zeros((unique_node_ids_num, NODE_FEAT_DIM))
        edge_raw_features = np.zeros((edge_ids_num, EDGE_FEAT_DIM))

        return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data
    else:
        print('To do')
        '''
        NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
        assert NODE_FEAT_DIM >= node_raw_features.shape[
            1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
        assert EDGE_FEAT_DIM >= edge_raw_features.shape[
            1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
        # padding the features of edges and nodes to the same dimension (172 for all the datasets)
        if node_raw_features.shape[1] < NODE_FEAT_DIM:
            node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
            node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
        if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
            edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
            edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)
    
        assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[
            1], 'Unaligned feature dimensions after feature padding!'
    
        # get the timestamp of validate and test set
        val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))
    
        src_node_ids = graph_df.u.values.astype(np.longlong)
        dst_node_ids = graph_df.i.values.astype(np.longlong)
        node_interact_times = graph_df.ts.values.astype(np.float64)
        edge_ids = graph_df.idx.values.astype(np.longlong)
        labels = graph_df.label.values
    
        # The setting of seed follows previous works
        random.seed(2025)
    
        train_mask = node_interact_times <= val_time
        val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
        test_mask = node_interact_times > test_time
    
        full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times,
                         edge_ids=edge_ids, labels=labels)
        train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                          node_interact_times=node_interact_times[train_mask],
                          edge_ids=edge_ids[train_mask], labels=labels[train_mask])
        val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                        node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask],
                        labels=labels[val_mask])
        test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                         node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask],
                         labels=labels[test_mask])
        '''

    # return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data
