import torch
import numpy as np
import torch.nn as nn
from collections import defaultdict

from utils.utils import NeighborSampler
from models.modules import TimeEncoder
from models.Fluxons.Fluxon import Fluxon
from models.Fluxons.FluxonRouter import FluxonRouter
from models.Fluxons.FluxonRouterCos import FluxonRouterCos
from models.Fluxons.FluxionUpdater import FluxonUpdater
from models.Fluxons.FluxionUpdaterCos import FluxonUpdaterCos
from models.modules import TimeEncoder
from torch.nn import MultiheadAttention
import torch.nn.functional as F
from models.Fluxons.TrendFeatureProjector import TrendFeatureProjector
# from models.MultiScaleDensityEncoder import MultiScaleDensityEncoder
from models.MultiScaleDensityEncoderv2 import MultiScaleDensityEncoder


class PopGroup(torch.nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray,
                 time_feat_dim: int, model_name: str = '', num_layers: int = 2,
                 num_heads: int = 2, dropout: float = 0.1,
                 device: str = 'cpu', item_nums: int = 0, user_nums: int = 0,
                 mode: str = "linear", k_select: int = 3, tau_start: float = 2.0,
                 tau_end: float = 0.5, total_steps: int = 1000, router_type: str = 'cosine',
                 distance_type: str = 'cosine', fluxion_init_type: str = 'zero', fluxion_ema: float = 0.3,
                 fluxion_size: int = 16, visual: bool = False):
        """
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param model_name: str, name of memory-based models, could be TGN, DyRep or JODIE
        :param num_layers: int, number of temporal graph convolution layers
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param device: str, device
        """
        super(PopGroup, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.device = device

        self.model_name = model_name
        # number of nodes, including the padded node
        self.num_nodes = self.node_raw_features.shape[0]
        self.num_items = item_nums
        self.memory_dim = self.node_feat_dim
        # since models use the identity function for message encoding, message dimension is 2 * memory_dim + time_feat_dim + edge_feat_dim
        self.message_dim = self.memory_dim + self.memory_dim + self.time_feat_dim + self.edge_feat_dim

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)

        # message module (models use the identity function for message encoding, hence, we only create MessageAggregator)
        self.message_aggregator = MessageAggregator()

        # memory modules
        self.memory_bank = MemoryBank(num_nodes=self.num_nodes, memory_dim=self.memory_dim)
        self.memory_updater = GRUMemoryUpdater(memory_bank=self.memory_bank, message_dim=self.message_dim,
                                               memory_dim=self.memory_dim)

        # slow memory modules
        # self.slow_memory_bank = nn.Parameter(torch.zeros((self.num_items, self.memory_dim), device=self.device), requires_grad=False)
        # self.slow_memory_updater = SlowMemoryUpdater(memory_bank=self.slow_memory_bank,
        #                                              memory_dim=self.memory_dim,
        #                                              mlp_hidden=2*self.memory_dim,
        #                                              device=self.device)
        # slow memory modules 2
        # self.lstm = LSTM_Predictor(input_dim=1)
        # slow memory modules 3
        num_types = 4
        self.type_emb = nn.Parameter(torch.randn(num_types, self.memory_dim))
        self.typecounter = TypeCountWeightedSum(memory_dim=self.memory_dim)
        self.lstm = LSTM_Predictor(input_dim=1, out_dim=self.memory_dim // 2 * 3)  # 1
        # MultiScaleDensityEncoder
        init_tau_days = np.array([1.0, 7.0, 30.0, 30.0 * 6], dtype=np.float64)
        self.density_encoder = MultiScaleDensityEncoder(
            self.memory_dim // 2,
            num_scales=4,
            device=self.device,
            use_mlp=True,
            log_compress=False,
            eps=1e-8,
            scale_div=86400.0,
            init_tau_days=init_tau_days
        )
        self.reduce_slow_memory_dim = nn.Linear(self.memory_dim * 2, self.memory_dim, bias=True).to(self.device)
        # self.reduce_slow_memory_dim = Reduce_SlowMem(self.memory_dim, device=self.device)

        self.num_users = user_nums + 1
        assert self.num_nodes == self.num_users + self.num_items, 'Data preprocessing error: dataset is inconsistent'

        # Fluxon modules
        self.mode = mode
        self.k_select = k_select
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.total_steps = total_steps
        self.router_type = router_type
        self.fluxion_init_type = fluxion_init_type
        self.num_fluxions = self.num_items // fluxion_size
        self.fluxon_bank = Fluxon(num_fluxons=self.num_fluxions,
                                  state_dim=self.memory_dim * 2,
                                  init_type=self.fluxion_init_type,
                                  device=self.device)
        if visual is True:
            self.fluxon_pop = torch.zeros(self.num_fluxions).to(self.device)
        if router_type == 'cosine':
            self.router = FluxonRouterCos(metric=distance_type, eps=1e-8)
        else:
            self.router = FluxonRouter(in_dim=self.memory_dim * 2,
                                       state_dim=self.memory_dim * 2,
                                       num_fluxons=self.num_fluxions,
                                       mode=self.mode,
                                       k_select=self.k_select,
                                       tau_start=self.tau_start,
                                       tau_end=self.tau_end,
                                       total_steps=self.total_steps,
                                       device=self.device)
        if router_type == 'cosine':
            self.fluxon_updater = FluxonUpdaterCos(in_dim=self.memory_dim * 2,
                                                   state_dim=self.memory_dim * 2,
                                                   ema_momentum=fluxion_ema,
                                                   device=self.device)
        else:
            self.fluxon_updater = FluxonUpdater(in_dim=self.memory_dim * 2,
                                                state_dim=self.memory_dim,
                                                ema_momentum=fluxion_ema,
                                                device=self.device)
        self.reduce_fluxion_dim = FluxionLinearReduce(in_dim=self.memory_dim * 2, out_dim=self.memory_dim, bias=False)

        # Fluxion aggregation modules
        self.history_len = 5
        # Use self.num_fluxions as the padding index, corresponding to an all-zero representation
        self.pad_flux_index = self.num_fluxions
        # Build per-item fluxion history: shape = (num_items, N), values in [0, self.num_fluxions]
        self.register_buffer(
            "item_flux_hist",
            torch.full((self.num_items, self.history_len), self.pad_flux_index, dtype=torch.long)
        )
        # Absolute time history: initialize with 0. We store N+1 timestamps to compute relative deltas correctly.
        self.register_buffer(
            "item_time_hist",
            torch.zeros(self.num_items, self.history_len + 1, dtype=torch.float32)
        )
        self.time_encoder_fluxion = TimeEncoder(time_dim=self.memory_dim * 2)
        self.transformers = nn.ModuleList([
            TransformerEncoder(attention_dim=self.memory_dim * 2, num_heads=self.num_heads,
                               dropout=self.dropout)
            for _ in range(self.num_layers)
        ])
        self.output_layer = nn.Linear(in_features=self.memory_dim * 2,
                                      out_features=self.memory_dim, bias=True)

    def compute_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                             node_interact_times: np.ndarray, edge_ids: np.ndarray,
                                             dst_slow_feature: np.ndarray, valid_index: np.ndarray,
                                             dst_sp_feature: np.ndarray, dst_time_feature: np.ndarray,
                                             labels=None, visual=False):
        """
        Compute source and destination node temporal embeddings.

        :param valid_index: indices with label (valid prediction edges)
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param edge_ids: ndarray, shape (batch_size, )
        :param dst_slow_feature: ndarray, shape (batch_size, 5, 7)
        :return:
        """
        # Tensor, shape (2 * batch_size, )
        node_ids = np.concatenate([src_node_ids, dst_node_ids])

        # compute new raw messages for source and destination nodes
        unique_src_node_ids, new_src_node_raw_messages = self.compute_new_node_raw_messages(src_node_ids=src_node_ids,
                                                                                            dst_node_ids=dst_node_ids,
                                                                                            node_interact_times=node_interact_times,
                                                                                            edge_ids=edge_ids)
        unique_dst_node_ids, new_dst_node_raw_messages = self.compute_new_node_raw_messages(src_node_ids=dst_node_ids,
                                                                                            dst_node_ids=src_node_ids,
                                                                                            node_interact_times=node_interact_times,
                                                                                            edge_ids=edge_ids)

        # store new raw messages for source and destination nodes
        self.memory_bank.store_node_raw_messages(node_ids=unique_src_node_ids, new_node_raw_messages=new_src_node_raw_messages)
        self.memory_bank.store_node_raw_messages(node_ids=unique_dst_node_ids, new_node_raw_messages=new_dst_node_raw_messages)

        assert edge_ids is not None
        # if the edges are positive, update the memories for source and destination nodes (since now we have new messages for them)
        self.update_memories(node_ids=node_ids, node_raw_messages=self.memory_bank.node_raw_messages)
        # clear raw messages for source and destination nodes since we have already updated the memory using them
        self.memory_bank.clear_node_raw_messages(node_ids=node_ids)
        fast_memory = self.memory_bank.get_memories(dst_node_ids)

        if valid_index[0].size == 0:
            # No prediction edges in this batch
            dst_node_embeddings = None
            return dst_node_embeddings

        # Below is the slow memory process

        # slow memory version3
        dst_sp_feature = torch.from_numpy(dst_sp_feature).float().to(self.device)  # [5,4]
        dst_sp_count = self.typecounter(self.type_emb, dst_sp_feature)
        slow_memory_counter = self.lstm(dst_sp_count)
        slow_memory_time = self.density_encoder(dst_time_feature, valid_index)
        slow_memory = self.reduce_slow_memory_dim(torch.concatenate((slow_memory_counter, slow_memory_time), dim=-1))

        # Below is the fluxion process
        # Prepare trend
        dst_trend_embeddings = torch.concatenate((fast_memory, slow_memory), dim=-1)
        valid_dst_trend_embeddings = dst_trend_embeddings[valid_index]
        valid_fast_memory = fast_memory[valid_index]
        valid_slow_memory = slow_memory[valid_index]
        assert torch.equal(valid_dst_trend_embeddings, torch.concatenate((valid_fast_memory, valid_slow_memory), dim=-1)), 'valid process error'
        # Routing
        A_states = self.fluxon_bank.get_all_fluxon().detach().clone()
        if self.router_type == 'cosine':
            idx = self.router(valid_dst_trend_embeddings, A_states)     # [B_valid, 1]
        else:
            idx, weight, tau = self.router(valid_dst_trend_embeddings, A_states)
        if visual is True:
            ind = idx.squeeze(1)
            labels = torch.from_numpy(labels[valid_index]).float().to(self.device)
            assert labels.shape == ind.shape, 'labels.shape != ind.shape'
            self.fluxon_pop[ind] = labels
        # Update to produce gradients
        if self.router_type == 'cosine':
            updated_fluxon_bank = self.fluxon_updater(valid_fast_memory, valid_slow_memory, idx, A_states)
        else:
            updated_fluxon_bank = self.fluxon_updater(valid_fast_memory, valid_slow_memory, idx, weight, A_states)
        # Write back (copy)
        with torch.no_grad():
            self.fluxon_bank.set_all_fluxon(None, updated_fluxon_bank)
        # Fetch selected fluxion embedding
        if self.router_type == 'cosine':
            sel = idx.squeeze(-1)   # [B_valid]
            fluxion_memory = updated_fluxon_bank[sel]   # [B_valid, D]
            fluxion_embedding = self.reduce_fluxion_dim(fluxion_memory)
        else:
            # Fetch k selected centers for each sample
            # idx: [B_valid, k] (LongTensor), weight: [B_valid, k]
            picked = updated_fluxon_bank[idx]  # [B_valid, k, D]
            # Weighted sum to get z
            fluxion_embedding = (weight.unsqueeze(-1) * picked).sum(dim=1)  # [B_valid, D]

        # Below is fluxion trajectory aggregation
        item_indices = dst_node_ids[valid_index] - self.num_users
        item_indices = torch.from_numpy(item_indices)   # [B_valid]
        flux_indices = idx.squeeze(-1)  # [B_valid]
        K, D = updated_fluxon_bank.shape
        pad_row = torch.zeros(1, D, device=self.device)  # [1, D]
        updated_fluxon_bank = torch.cat([updated_fluxon_bank, pad_row], dim=0)  # [K+1, D]
        node_interact_times_all = torch.from_numpy(node_interact_times).float().to(self.device)  # [B]
        event_times = node_interact_times_all[valid_index]  # [B_valid]
        # Call trajectory aggregation function to get trajectory embedding
        traj_emb = self.aggregate_fluxion_trajectory(
            item_indices=item_indices,            # [B_valid]
            flux_indices=flux_indices,            # [B_valid]
            updated_fluxon_bank=updated_fluxon_bank,  # [K+1, D], last row is dummy=0
            event_times=event_times               # [B_valid]
        )

        if visual:
            dst_node_embeddings = fluxion_embedding
        else:
            dst_node_embeddings = torch.concatenate(
                (valid_fast_memory, valid_slow_memory, fluxion_embedding, traj_emb), dim=-1
            )

        return dst_node_embeddings

    def update_memories(self, node_ids: np.ndarray, node_raw_messages: dict):
        """
        Update memories for nodes in node_ids.
        :param node_ids: ndarray, shape (num_nodes, )
        :param node_raw_messages: dict, dictionary of list, {node_id: list of tuples},
            each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        :return:
        """
        unique_node_ids, unique_node_messages, unique_node_timestamps = self.message_aggregator.aggregate_messages(
            node_ids=node_ids,
            node_raw_messages=node_raw_messages
        )

        self.memory_updater.update_memories(unique_node_ids=unique_node_ids, unique_node_messages=unique_node_messages,
                                            unique_node_timestamps=unique_node_timestamps)

    def compute_new_node_raw_messages(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                      node_interact_times: np.ndarray, edge_ids: np.ndarray):
        """
        Compute new raw messages for nodes in src_node_ids.
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param edge_ids: ndarray, shape (batch_size, )
        :return:
        """
        src_node_memories = self.memory_bank.get_memories(node_ids=src_node_ids)
        dst_node_memories = self.memory_bank.get_memories(node_ids=dst_node_ids)

        src_node_delta_times = torch.from_numpy(node_interact_times).float().to(self.device) - \
                               self.memory_bank.node_last_updated_times[torch.from_numpy(src_node_ids)]
        src_node_delta_time_features = self.time_encoder(src_node_delta_times.unsqueeze(dim=1)).reshape(len(src_node_ids), -1)

        edge_features = self.edge_raw_features[torch.from_numpy(edge_ids)]

        new_src_node_raw_messages = torch.cat([src_node_memories, dst_node_memories, src_node_delta_time_features, edge_features], dim=1)

        new_node_raw_messages = defaultdict(list)
        unique_node_ids = np.unique(src_node_ids)

        for i in range(len(src_node_ids)):
            new_node_raw_messages[src_node_ids[i]].append((new_src_node_raw_messages[i], node_interact_times[i]))

        return unique_node_ids, new_node_raw_messages

    def aggregate_fluxion_trajectory(
            self,
            item_indices: torch.Tensor,          # [B_valid] item row indices (0..num_items-1)
            flux_indices: torch.Tensor,          # [B_valid] fluxion index routed this time
            updated_fluxon_bank: torch.Tensor,   # [K+1, D] latest fluxion memory
            event_times: torch.Tensor            # [B_valid] absolute event time (float)
    ) -> torch.Tensor:
        """
        Aggregate each item's historical fluxion trajectory + relative time deltas with a Transformer.

        History buffers:
          - item_flux_hist: [num_items, N]
            Flux indices of the most recent N events.
            Position 0 is the most recent, N-1 is the oldest.
            pad_flux_index indicates invalid (padding).
          - item_time_hist: [num_items, N+1]
            Absolute times for the most recent N+1 events.
            Position 0 corresponds to item_flux_hist[:,0],
            1 corresponds to flux_hist[:,1], ...,
            N corresponds to one event earlier than the oldest flux (to compute the last interval).

        Relative time deltas:
          Δt_i = t_i - t_{i+1}, i = 0..N-1
          Each Δt_i corresponds to flux_hist[:, i] (gap between this event and the previous one).

        Return:
          traj_emb: [B_valid, D] aggregated trajectory representation
        """
        K = int(self.num_fluxions)
        N = int(self.history_len)
        item_indices = item_indices.to(device=self.device, dtype=torch.long)      # [B_valid]
        flux_indices = flux_indices.to(device=self.device, dtype=torch.long)      # [B_valid]
        event_times = event_times.to(device=self.device, dtype=torch.float32)     # [B_valid]
        B_valid = item_indices.size(0)
        assert flux_indices.shape == (B_valid,), f"flux_indices should be [B_valid], got {flux_indices.shape}"
        assert event_times.shape == (B_valid,), f"event_times should be [B_valid], got {event_times.shape}"

        # 1) Fetch history for the batch from buffers
        hist_flux = self.item_flux_hist[item_indices]  # [B_valid, N]
        hist_time = self.item_time_hist[item_indices]  # [B_valid, N+1]

        # 2) Shift history by one position and insert the new event at the front
        hist_flux = torch.roll(hist_flux, shifts=1, dims=1)
        hist_time = torch.roll(hist_time, shifts=1, dims=1)
        hist_flux[:, 0] = flux_indices
        hist_time[:, 0] = event_times

        # 3) Write updated history back to buffers
        self.item_flux_hist[item_indices] = hist_flux
        self.item_time_hist[item_indices] = hist_time

        # 4) Gather fluxion memory for each history position
        hist_flux_mem = updated_fluxon_bank[hist_flux]  # [B_valid, N, D]

        # 5) Compute time deltas (using N+1 timestamps to derive N intervals)
        assert torch.all(hist_time[:, :-1] >= hist_time[:, 1:]), "hist_time is not sorted in descending order"

        ref_t = hist_time[:, 0:1]
        t_targets = hist_time[:, :N]
        dt = (ref_t - t_targets).clamp_min(0.0)  # [B_valid, N]

        # Time encoding
        time_emb = self.time_encoder_fluxion(dt)  # [B_valid, N, time_dim=D]

        seq = hist_flux_mem + time_emb  # [B_valid, N, D]
        for transformer in self.transformers:
            seq = transformer(seq)  # [B_valid, N, D]

        traj_emb = seq[:, 0, :]         # [B_valid, D]
        traj_emb = self.output_layer(traj_emb)  # [B_valid, D/2]
        return traj_emb

    def reset_item_history(self):
        """
        Call at the beginning of each epoch to clear per-item fluxion and time histories.
        """
        # Fill flux trajectory with pad_flux_index
        self.item_flux_hist.fill_(self.pad_flux_index)
        # Reset time trajectory to 0
        self.item_time_hist.zero_()


# Message-related Modules
class MessageAggregator(nn.Module):

    def __init__(self):
        """
        Message aggregator. Given a batch of node ids and corresponding messages,
        aggregate messages with the same node id.
        """
        super(MessageAggregator, self).__init__()

    def aggregate_messages(self, node_ids: np.ndarray, node_raw_messages: dict):
        """
        Given a list of node ids and corresponding messages, aggregate messages with the same node id
        (only keep the last message for each node).

        :param node_ids: ndarray, shape (batch_size, )
        :param node_raw_messages: dict, dictionary of list, {node_id: list of tuples},
            each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        :return:
        """
        unique_node_ids = np.unique(node_ids)
        unique_node_messages, unique_node_timestamps, to_update_node_ids = [], [], []

        for node_id in unique_node_ids:
            if len(node_raw_messages[node_id]) > 0:
                to_update_node_ids.append(node_id)
                unique_node_messages.append(node_raw_messages[node_id][-1][0])
                unique_node_timestamps.append(node_raw_messages[node_id][-1][1])

        to_update_node_ids = np.array(to_update_node_ids)
        unique_node_messages = torch.stack(unique_node_messages, dim=0) if len(unique_node_messages) > 0 else torch.Tensor([])
        unique_node_timestamps = np.array(unique_node_timestamps)

        return to_update_node_ids, unique_node_messages, unique_node_timestamps


# Memory-related Modules
class MemoryBank(nn.Module):

    def __init__(self, num_nodes: int, memory_dim: int):
        """
        Memory bank stores node memories, last updated times, and raw messages.

        :param num_nodes: int, number of nodes
        :param memory_dim: int, dimension of node memories
        """
        super(MemoryBank, self).__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim

        self.node_memories = nn.Parameter(torch.zeros((self.num_nodes, self.memory_dim)), requires_grad=False)
        self.node_last_updated_times = nn.Parameter(torch.zeros(self.num_nodes), requires_grad=False)
        self.node_raw_messages = defaultdict(list)

        self.__init_memory_bank__()

    def __init_memory_bank__(self):
        """
        Initialize all memories and node_last_updated_times to zero vectors, and reset node_raw_messages.
        Should be called at the start of each epoch.
        """
        self.node_memories.data.zero_()
        self.node_last_updated_times.data.zero_()
        self.node_raw_messages = defaultdict(list)

    def get_memories(self, node_ids: np.ndarray):
        """
        Get memories for nodes in node_ids.
        :param node_ids: ndarray, shape (batch_size, )
        :return:
        """
        return self.node_memories[torch.from_numpy(node_ids)]

    def set_memories(self, node_ids: np.ndarray, updated_node_memories: torch.Tensor):
        """
        Set memories for nodes in node_ids to updated_node_memories.
        :param node_ids: ndarray, shape (batch_size, )
        :param updated_node_memories: Tensor, shape (num_unique_node_ids, memory_dim)
        :return:
        """
        self.node_memories[torch.from_numpy(node_ids)] = updated_node_memories

    def backup_memory_bank(self):
        """
        Backup the memory bank: return copies of current memories, last updated times, and raw messages.
        :return:
        """
        cloned_node_raw_messages = {}
        for node_id, node_raw_messages in self.node_raw_messages.items():
            cloned_node_raw_messages[node_id] = [(node_raw_message[0].clone(), node_raw_message[1].copy())
                                                 for node_raw_message in node_raw_messages]

        return self.node_memories.data.clone(), self.node_last_updated_times.data.clone(), cloned_node_raw_messages

    def reload_memory_bank(self, backup_memory_bank: tuple):
        """
        Reload the memory bank from a backup tuple (node_memories, node_last_updated_times, node_raw_messages).
        :param backup_memory_bank: tuple
        :return:
        """
        self.node_memories.data, self.node_last_updated_times.data = backup_memory_bank[0].clone(), backup_memory_bank[1].clone()

        self.node_raw_messages = defaultdict(list)
        for node_id, node_raw_messages in backup_memory_bank[2].items():
            self.node_raw_messages[node_id] = [(node_raw_message[0].clone(), node_raw_message[1].copy())
                                               for node_raw_message in node_raw_messages]

    def detach_memory_bank(self):
        """
        Detach gradients of node memories and stored raw messages.
        """
        self.node_memories.detach_()

        for node_id, node_raw_messages in self.node_raw_messages.items():
            new_node_raw_messages = []
            for node_raw_message in node_raw_messages:
                new_node_raw_messages.append((node_raw_message[0].detach(), node_raw_message[1]))
            self.node_raw_messages[node_id] = new_node_raw_messages

    def store_node_raw_messages(self, node_ids: np.ndarray, new_node_raw_messages: dict):
        """
        Store raw messages for nodes in node_ids.
        :param node_ids: ndarray, shape (batch_size, )
        :param new_node_raw_messages: dict, dictionary of list, {node_id: list of tuples},
            each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        :return:
        """
        for node_id in node_ids:
            self.node_raw_messages[node_id].extend(new_node_raw_messages[node_id])

    def clear_node_raw_messages(self, node_ids: np.ndarray):
        """
        Clear raw messages for nodes in node_ids.
        :param node_ids: ndarray, shape (batch_size, )
        :return:
        """
        for node_id in node_ids:
            self.node_raw_messages[node_id] = []

    def get_node_last_updated_times(self, unique_node_ids: np.ndarray):
        """
        Get last updated times for nodes in unique_node_ids.
        :param unique_node_ids: ndarray, (num_unique_node_ids, )
        :return:
        """
        return self.node_last_updated_times[torch.from_numpy(unique_node_ids)]

    def extra_repr(self):
        """
        Set extra representation of the module (customized print info).
        """
        return 'num_nodes={}, memory_dim={}'.format(self.node_memories.shape[0], self.node_memories.shape[1])


class MemoryUpdater(nn.Module):

    def __init__(self, memory_bank: MemoryBank):
        """
        Memory updater.
        :param memory_bank: MemoryBank
        """
        super(MemoryUpdater, self).__init__()
        self.memory_bank = memory_bank

    def update_memories(self, unique_node_ids: np.ndarray, unique_node_messages: torch.Tensor,
                        unique_node_timestamps: np.ndarray):
        """
        Update memories for nodes in unique_node_ids.
        :param unique_node_ids: ndarray, shape (num_unique_node_ids, )
        :param unique_node_messages: Tensor, shape (num_unique_node_ids, message_dim)
        :param unique_node_timestamps: ndarray, shape (num_unique_node_ids, )
        :return:
        """
        if len(unique_node_ids) <= 0:
            return

        assert (self.memory_bank.get_node_last_updated_times(unique_node_ids) <=
                torch.from_numpy(unique_node_timestamps).float().to(unique_node_messages.device)).all().item(), \
            "Trying to update memory to time in the past!"

        node_memories = self.memory_bank.get_memories(node_ids=unique_node_ids)
        updated_node_memories = self.memory_updater(unique_node_messages, node_memories)
        self.memory_bank.set_memories(node_ids=unique_node_ids, updated_node_memories=updated_node_memories)

        self.memory_bank.node_last_updated_times[torch.from_numpy(unique_node_ids)] = \
            torch.from_numpy(unique_node_timestamps).float().to(unique_node_messages.device)

    def get_updated_memories(self, unique_node_ids: np.ndarray, unique_node_messages: torch.Tensor,
                             unique_node_timestamps: np.ndarray):
        """
        Compute updated memories based on inputs (for computation only), without writing back to the memory bank.
        :param unique_node_ids: ndarray, shape (num_unique_node_ids, )
        :param unique_node_messages: Tensor, shape (num_unique_node_ids, message_dim)
        :param unique_node_timestamps: ndarray, shape (num_unique_node_ids, )
        :return:
        """
        if len(unique_node_ids) <= 0:
            return self.memory_bank.node_memories.data.clone(), self.memory_bank.node_last_updated_times.data.clone()

        assert (self.memory_bank.get_node_last_updated_times(unique_node_ids=unique_node_ids) <=
                torch.from_numpy(unique_node_timestamps).float().to(unique_node_messages.device)).all().item(), \
            "Trying to update memory to time in the past!"

        updated_node_memories = self.memory_bank.node_memories.data.clone()
        updated_node_memories[torch.from_numpy(unique_node_ids)] = self.memory_updater(
            unique_node_messages,
            updated_node_memories[torch.from_numpy(unique_node_ids)]
        )

        updated_node_last_updated_times = self.memory_bank.node_last_updated_times.data.clone()
        updated_node_last_updated_times[torch.from_numpy(unique_node_ids)] = \
            torch.from_numpy(unique_node_timestamps).float().to(unique_node_messages.device)

        return updated_node_memories, updated_node_last_updated_times


class GRUMemoryUpdater(MemoryUpdater):

    def __init__(self, memory_bank: MemoryBank, message_dim: int, memory_dim: int):
        """
        GRU-based memory updater.
        :param memory_bank: MemoryBank
        :param message_dim: int, dimension of node messages
        :param memory_dim: int, dimension of node memories
        """
        super(GRUMemoryUpdater, self).__init__(memory_bank)
        self.memory_updater = nn.GRUCell(input_size=message_dim, hidden_size=memory_dim)


class RNNMemoryUpdater(MemoryUpdater):

    def __init__(self, memory_bank: MemoryBank, message_dim: int, memory_dim: int):
        """
        RNN-based memory updater.
        :param memory_bank: MemoryBank
        :param message_dim: int, dimension of node messages
        :param memory_dim: int, dimension of node memories
        """
        super(RNNMemoryUpdater, self).__init__(memory_bank)
        self.memory_updater = nn.RNNCell(input_size=message_dim, hidden_size=memory_dim)


class SlowMemoryUpdater(nn.Module):
    """
    Update slow_memory_old using dst_slow_feature and return the updated slow memory.
    - Inputs:
        slow_memory_old: (B, D)
        dst_slow_feature: (B, 7)
        dst_node_ids: (B,) LongTensor
    - Process:
        1) dst_slow_feature -> MLP -> (B, D)
        2) GRUCell(input=(B, D), hidden=slow_memory_old) -> updated (B, D)
        3) The caller can write updated back to slow_memory_bank[dst_node_ids] if needed.
           * If dst_node_ids has duplicates, use the last occurrence.
    - Output:
        updated: (B, D)  # per-sample updated memory aligned with slow_memory_old
    """
    def __init__(self, memory_bank: nn.Parameter, memory_dim: int, mlp_hidden: int = 128, device: str = 'cpu'):
        super().__init__()
        assert isinstance(memory_bank, nn.Parameter), "memory_bank must be an nn.Parameter"
        self.memory_bank = memory_bank
        self.memory_dim = memory_dim
        self.device = device
        self.mlp = nn.Sequential(
            nn.Linear(7, mlp_hidden),
            nn.ReLU(inplace=False),
            nn.Linear(mlp_hidden, memory_dim),
        ).to(self.device)
        self.gru = nn.GRUCell(input_size=memory_dim, hidden_size=memory_dim).to(self.device)

    def forward(self, slow_memory_old: torch.Tensor,
                dst_slow_feature: torch.Tensor,
                dst_node_ids: torch.Tensor) -> torch.Tensor:
        """
        slow_memory_old: (B, D)
        dst_slow_feature: (B, 7)
        dst_node_ids: (B,) LongTensor (convert numpy to torch outside if needed)
        """
        assert slow_memory_old.dim() == 2 and slow_memory_old.size(1) == self.memory_dim
        assert dst_slow_feature.shape[0] == slow_memory_old.shape[0] and dst_slow_feature.shape[1] == 7
        assert dst_node_ids.shape[0] == slow_memory_old.shape[0]

        inp = self.mlp(dst_slow_feature)     # (B, D)
        updated = self.gru(inp, slow_memory_old)  # (B, D)
        return updated


class FluxionLinearReduce(nn.Module):
    """A single linear layer: in_dim -> out_dim (reduce fluxion embedding to half)."""
    def __init__(self, in_dim: int, out_dim: int, bias: bool = False):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=bias)
        if bias:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Reduce_SlowMem(nn.Module):
    def __init__(self, memory_dim, hidden_dim=None, bias=True, device=None):
        super(Reduce_SlowMem, self).__init__()
        if hidden_dim is None:
            hidden_dim = memory_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(memory_dim * 2, hidden_dim, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_dim, memory_dim, bias=bias)
        )
        self.device = device
        if device is not None:
            self.to(device)

    def forward(self, x):
        # x shape: (..., memory_dim * 2)
        return self.mlp(x)


class TransformerEncoder(nn.Module):
    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Transformer encoder.
        :param attention_dim: int, dimension of the attention vector
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(TransformerEncoder, self).__init__()
        self.multi_head_attention = MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=attention_dim, out_features=4 * attention_dim),
            nn.Linear(in_features=4 * attention_dim, out_features=attention_dim)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(attention_dim),
            nn.LayerNorm(attention_dim)
        ])

    def forward(self, inputs: torch.Tensor):
        """
        Encode inputs with a Transformer encoder.
        :param inputs: Tensor, shape (batch_size, num_patches, attention_dim)
        :return:
        """
        transposed_inputs = inputs.transpose(0, 1)
        transposed_inputs = self.norm_layers[0](transposed_inputs)
        hidden_states = self.multi_head_attention(query=transposed_inputs, key=transposed_inputs, value=transposed_inputs)[0].transpose(0, 1)
        outputs = inputs + self.dropout(hidden_states)
        hidden_states = self.linear_layers[1](self.dropout(F.gelu(self.linear_layers[0](self.norm_layers[1](outputs)))))
        outputs = outputs + self.dropout(hidden_states)
        return outputs


class LSTM_Predictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, out_dim: int = 172, num_layers: int = 1, dropout: float = 0.1):
        """
        LSTM-based predictor for sequential features.
        :param input_dim: int, feature dimension per time step
        :param hidden_dim: int, hidden dimension of LSTM
        :param num_layers: int, number of LSTM layers
        :param dropout: float, dropout rate
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        :param x: Tensor, shape (batch_size, seq_len, input_dim)
        :return: Tensor, shape (batch_size, out_dim)
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        return self.fc(out)


class TypeCountWeightedSum(nn.Module):
    """
    x:   [B, 5, 4]   (each of the 4 dims is the count of one type)
    out: [B, 5, 1]   (a scalar per window)
    """
    def __init__(self, memory_dim: int = 32):
        super().__init__()
        # Learnable mapping: D -> 1
        self.proj = nn.Linear(memory_dim, 1, bias=False)

    def forward(self, type_emb: torch.Tensor, x: torch.Tensor):
        """
        x: [B, 5, 4]
        """
        # 1) Scalar weights for each type
        #    w: [4]
        w = self.proj(type_emb).squeeze(-1)

        # 2) Weighted sum
        out = torch.einsum("btk,k->bt", x.float(), w)
        return out.unsqueeze(-1)
        # Alternative:
        # out = torch.einsum("btk,kd->btd", x.float(), type_emb)
        # return out
