import torch
import torch.nn as nn


class DEP(nn.Module):
    """
    Data-driven Edge Pruning (DEP) for fully connected FC brain graphs.

    One learnable score matrix W (N x N), shared by all subjects.

        S = sigmoid(W + W^T)          symmetric edge scores in (0, 1)
        H = hard 0/1 mask             prune the lowest `min_sp` fraction of undirected edges (by S)
        M = H + S - S.detach()        straight-through estimator:
                                      forward  -> exact 0/1 mask H
                                      backward -> gradient of the loss w.r.t. the scores S

    One mask H decides which directed edges remain. By default, the same mask also
    masks node features; set ``params.mask_node_features`` to ``False`` when node
    features and edge attributes represent different information:
        edges          (i, j) pruned  ->  removed from edge_index, together with its edge_attr
                       (i, j) kept    ->  stays, with weight FC_ij * M[i, j]  (= FC_ij in the forward pass)
        node features  x_i = FC row i -> optionally x_i[j] = FC_ij * M[i, j]
    When node features are masked, both channels are read from the same H at the same
    (i, j). H is symmetric, so (i, j) and (j, i) are always removed together.

    Pruned edges leave the graph, but their scores still receive a task gradient through the node
    features, so a pruned connection can come back if the task needs it.

    Expected input (per batch, all graphs fully connected):
        data.x          [B*N, N]  node features = FC rows, column j = node j (same node order as edge_index)
        data.edge_attr  [E] or [E, 1]  FC weight of each directed edge
        data.edge_index, data.batch, data.ptr  (standard PyG batch)
    """

    def __init__(self, params, input_dim, output_dim):
        super().__init__()
        self.alpha = params.alpha              # L1 weight on the scores
        self.beta = params.beta                # L2 weight on the scores
        self.min_sp = params.curr_sp           # fraction of undirected edges currently pruned
        self.mask_node_features = getattr(params, "mask_node_features", True)
        self.N = input_dim                     # number of nodes (= number of node features, FC rows)
        self.output_dim = output_dim

        self.weight_mask = nn.Parameter(torch.empty(self.N, self.N))
        self.register_buffer("iu", torch.triu_indices(self.N, self.N, offset=1), persistent=False)

        self.sym_weight_mask = None            # S from the last forward pass
        self.edge_mask = None                  # H from the last forward pass, bool [N, N], True = kept
        self.fixed_mask = None                 # optional bool [N, N]: use a given mask (controls)

        # consistency checks: the input check runs on the first batch when node features
        # are coupled to the edge mask; the output check
        # (expensive) runs only when debug=True
        self._input_checked = False
        self.debug = getattr(params, "debug_dep", False)

        self.reset_parameters()

    # ------------------------------------------------------------------ setup
    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight_mask)

    # ------------------------------------------------------------------ masks
    def scores(self):
        """Symmetric edge scores S in (0, 1)."""
        return torch.sigmoid(self.weight_mask + self.weight_mask.T)

    @torch.no_grad()
    def hard_mask(self, S=None):
        """Keep all edges except the lowest `min_sp` fraction (ranked on the upper triangle).
        Built on the upper triangle and mirrored, so (i, j) and (j, i) are always kept or pruned together."""
        if S is None:
            S = self.scores()
        iu = self.iu.to(S.device)
        s = S[iu[0], iu[1]]                                    # one score per undirected edge
        keep = torch.ones_like(s, dtype=torch.bool)
        n_prune = int(round(s.numel() * self.min_sp))
        if n_prune > 0:
            keep[torch.argsort(s)[:n_prune]] = False
        H = torch.zeros(self.N, self.N, dtype=torch.bool, device=S.device)
        H[iu[0], iu[1]] = keep
        H = H | H.T
        H.fill_diagonal_(True)                                 # diagonal entry of each FC row is not an edge
        return H

    def set_fixed_mask(self, H):
        """Use a given bool [N, N] mask instead of the learned one (random / top-k / retrained-DEP controls).
        Pass None to go back to the learned mask."""
        if H is None:
            self.fixed_mask = None
            return
        H = torch.as_tensor(H, dtype=torch.bool, device=self.weight_mask.device)
        H = H | H.T
        H.fill_diagonal_(True)
        self.fixed_mask = H

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _local_pairs(data):
        """For every edge in the batch: its graph g and its local (i, j) = (source, target) in 0..N-1."""
        src, dst = data.edge_index
        g = data.batch[src]
        return g, src - data.ptr[g], dst - data.ptr[g]

    @torch.no_grad()
    def _check_input(self, data, B, N):
        """Edge weights and node features must describe the same matrix in the same node order:
        edge (i, j) of graph g must carry X_g[i, j]. If this fails, masking edges and features
        with the same H would NOT remove the same connection from both."""
        g, i, j = self._local_pairs(data)
        X = data.x.view(B, N, N)
        w = data.edge_attr.reshape(-1).to(X.dtype)
        if not torch.allclose(X[g, i, j], w, atol=1e-5, rtol=1e-4):
            raise ValueError(
                "Edge weights do not match the node features: edge (i, j) does not carry X[i, j]. "
                "Either the node order differs between x and edge_index, or edge_attr is not the same "
                "FC matrix as the node features. DEP cannot prune both channels consistently."
            )
        if (i == j).any():
            print("DEP: self-loops present in edge_index; they are always kept (diagonal of H).")

    @torch.no_grad()
    def _check_output(self, out, B, N, H):
        """After pruning: every remaining edge is kept in H and carries the (unchanged) X[i, j];
        every pruned pair is absent from edge_index and zero in the features."""
        g, i, j = self._local_pairs(out)
        assert H[i, j].all(), "a pruned edge is still in edge_index"
        X = out.x.view(B, N, N)
        w = out.edge_attr.reshape(-1).to(X.dtype)
        if self.mask_node_features:
            assert torch.allclose(X[g, i, j], w, atol=1e-5, rtol=1e-4), "edge weight != feature entry"
        off = ~torch.eye(N, dtype=torch.bool, device=H.device)
        if self.mask_node_features:
            assert (X[:, ~H & off] == 0).all(), "a pruned pair is non-zero in the node features"
        n_kept_pairs = int((H & off).sum())                     # directed pairs kept per graph
        assert out.edge_index.size(1) == B * n_kept_pairs or (i == j).any(), \
            "number of remaining edges does not match the mask"

    # ---------------------------------------------------------------- forward
    def forward(self, data):
        """
        Returns:
            out : a pruned copy of `data` (the input batch is never modified)
            H   : bool [N, N] mask used in this pass (True = kept)
        """
        B, N = data.num_graphs, self.N
        if data.x.dim() != 2 or data.x.size(0) != B * N or data.x.size(1) != N:
            raise ValueError(
                f"DEP expects node features = FC rows, x of shape [B*N, N] = [{B * N}, {N}], "
                f"got {tuple(data.x.shape)}. Check that the dataset is the FC-row (PearC) one."
            )
        if not self._input_checked and self.mask_node_features:
            self._check_input(data, B, N)
            self._input_checked = True

        if self.fixed_mask is not None:
            H = self.fixed_mask.to(data.x.device)
            M = H.to(data.x.dtype)                             # no gradient to the scores
            self.sym_weight_mask = self.scores()               # keeps l1/l2 well defined (not used for the mask)
        else:
            S = self.scores()
            self.sym_weight_mask = S
            H = self.hard_mask(S)
            M = H.to(S.dtype) + S - S.detach()                 # straight-through estimator
        self.edge_mask = H

        out = data.clone()

        # (1) Optionally mask node features with the same mask as the edges.
        if self.mask_node_features:
            out.x = (data.x.view(B, N, N) * M.unsqueeze(0)).view(B * N, N)
        else:
            out.x = data.x.clone()

        # (2) edges: look up the SAME mask at the SAME (i, j) for every edge (src=i, dst=j)
        g, i, j = self._local_pairs(data)
        keep_e = H[i, j]                                       # bool [E]; edge order is preserved
        w = data.edge_attr.reshape(-1) * M[i, j]               # = FC_ij forward; gradient to S backward
        if data.edge_attr.dim() == 2:
            w = w.unsqueeze(-1)

        out.edge_index = data.edge_index[:, keep_e]
        out.edge_attr = w[keep_e]

        # any other per-edge attribute is filtered the same way, so all edge tensors stay aligned
        edge_attribute_keys = ["edge_index", "edge_attr"]
        for key, value in data:
            if key in ("edge_index", "edge_attr") or not torch.is_tensor(value):
                continue
            if data.is_edge_attr(key) and value.size(0) == keep_e.numel():
                out[key] = value[keep_e]
                edge_attribute_keys.append(key)

        # A cloned PyG Batch retains the original edge slice boundaries. Update them
        # after filtering so sampled[graph_id] reconstructs each graph correctly.
        slice_dict = getattr(out._store, "_slice_dict", None)
        if slice_dict is not None and "edge_index" in slice_dict:
            slice_dict = out._store._slice_dict = dict(slice_dict)
            kept_per_graph = torch.bincount(g[keep_e], minlength=B).cpu()
            edge_slices = torch.cat(
                (
                    torch.zeros(1, dtype=torch.long),
                    torch.cumsum(kept_per_graph, dim=0),
                )
            )
            for key in edge_attribute_keys:
                if key in slice_dict:
                    slice_dict[key] = edge_slices

        if self.debug:
            self._check_output(out, B, N, H)

        return out, H

    # ---------------------------------------------------------- pruning level
    def update_prune(self, sp_increase):
        self.min_sp = round(self.min_sp + sp_increase, 5)

    # ---------------------------------------------------------- regularisers
    def l1_norm(self):
        return self.alpha * self.sym_weight_mask.abs().sum()

    def l2_norm(self):
        return self.beta * self.sym_weight_mask.pow(2).sum()

    # ---------------------------------------------------------------- getters
    def get_params(self):
        return self.sym_weight_mask

    def get_edge_mask(self):
        """bool [N, N] from the last forward pass (True = kept)."""
        return self.edge_mask

    @torch.no_grad()
    def get_edge_vector(self):
        """Current mask as bool [N*(N-1)/2] in upper-triangle row-major order
        (same order as np.triu_indices(N, 1), i.e. the LR / top-k / MoG files)."""
        H = self.fixed_mask if self.fixed_mask is not None else self.hard_mask()
        iu = self.iu.to(H.device)
        return H[iu[0], iu[1]].cpu().numpy()
