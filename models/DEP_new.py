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

    The mask is applied to BOTH channels that carry FC information:
        node features  x_i  = FC row i   ->  x_i * M[i, :]   (a pruned connection leaves the features)
        edge weights   w_ij = FC_ij      ->  w_ij * M[i, j]  (a pruned edge gets weight 0)
    so the GNN has no access to a pruned connection.

    Expected input (per batch, all graphs fully connected, same edge order):
        data.x          [B*N, N]  node features = FC rows
        data.edge_attr  [E] or [E, 1]  FC weight of each directed edge
        data.edge_index, data.batch, data.ptr  (standard PyG batch)
    """

    def __init__(self, params, input_dim, output_dim):
        super().__init__()
        self.alpha = params.alpha              # L1 weight on the scores
        self.beta = params.beta                # L2 weight on the scores
        self.min_sp = params.curr_sp           # fraction of undirected edges currently pruned
        self.N = input_dim                     # number of nodes (= number of node features, FC rows)
        self.output_dim = output_dim

        self.weight_mask = nn.Parameter(torch.empty(self.N, self.N))
        self.register_buffer("iu", torch.triu_indices(self.N, self.N, offset=1), persistent=False)

        self.sym_weight_mask = None            # S from the last forward pass
        self.edge_mask = None                  # H from the last forward pass, bool [N, N], True = kept
        self.fixed_mask = None                 # optional bool [N, N]: use a given mask (controls)

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
        s = S[self.iu[0], self.iu[1]]                          # one score per undirected edge
        keep = torch.ones_like(s, dtype=torch.bool)
        n_prune = int(round(s.numel() * self.min_sp))
        if n_prune > 0:
            keep[torch.argsort(s)[:n_prune]] = False
        H = torch.zeros(self.N, self.N, dtype=torch.bool, device=S.device)
        H[self.iu[0], self.iu[1]] = keep
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

        # (1) node features: pruned FC rows
        out.x = (data.x.view(B, N, N) * M.unsqueeze(0)).view(B * N, N)

        # (2) edge weights: pruned FC weights (pruned edges stay in edge_index with weight 0,
        #     so their scores still receive a gradient and can grow back)
        src, dst = data.edge_index
        g = data.batch[src]
        ls = src - data.ptr[g]                                 # local node index 0..N-1
        ld = dst - data.ptr[g]
        out.edge_attr = data.edge_attr.reshape(-1) * M[ls, ld]

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
        return H[self.iu[0], self.iu[1]].cpu().numpy()
