import torch
import torch.nn as nn
from torch_geometric.nn import GraphConv


class PrimalGNN(nn.Module):
    '''
    Features
    [Demand, MaxProd, FlowUB, FlowLB, LocNode, ProdNode, LineNode]

    Output:
    - ProdNode: generation at each producer node
    - LineNode: flow on each line

    Unmet Demand will be computed based on predicted generation, predicted flow and demand.
    '''
    
    def __init__(self, node_to_gen_mask, line_flow_mask, n_loc, n_gen, n_line, in_dim=7, hidden_dim=16, num_layers=3):
        super().__init__()
        self.encoder = torch.nn.Linear(in_dim, hidden_dim)
        print("Hidden dim in PrimalGNN:", hidden_dim)
        print(f"Number of loc nodes: {n_loc}, gen nodes: {n_gen}, line nodes: {n_line}")
        self.convs = torch.nn.ModuleList([
            GraphConv(in_channels=hidden_dim, out_channels = hidden_dim) for _ in range(num_layers)
        ])

        # Heads
        self.prod_head = torch.nn.Linear(hidden_dim, 1)   # only for ProdNode==1
        self.flow_head = torch.nn.Linear(hidden_dim, 1)   # only for LineNode==1

        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.node_to_gen_mask = torch.tensor(node_to_gen_mask, dtype=torch.float64) # (|N|, |G|)
        self.line_flow_mask = torch.tensor(line_flow_mask, dtype = torch.float64)      # (|N|, |F|)

        self.L = n_loc
        self.G = n_gen
        self.F = n_line
        self.N = n_loc + n_gen + n_line

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize nn.Linear layers (encoder, heads)
        for m in [self.encoder, self.prod_head, self.flow_head]:
            # Kaiming Uniform initialization for weights, suitable for ReLU activation
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            # Initialize biases to zero
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        # Explicitly reset parameters for GraphConv layers (uses PyG's optimized initialization)
        for conv in self.convs:
            if hasattr(conv, 'reset_parameters'):
                 conv.reset_parameters()

        

    def compute_unmet_demand(self, pred_p, pred_f, demand):
        """  
        Given:
        pred_p: [B, |G|] predicted generation at producer nodes
        pred_f: [B, |F|] predicted flow at line nodes
        demand: [B, |N|] demand at location nodes

        Output:
        MD: [B, |N|] unmet demand at location nodes
        """
        combined_flow = torch.matmul(pred_p, self.node_to_gen_mask.T) + torch.matmul(pred_f, self.line_flow_mask.T)

        md = demand - combined_flow
        return md

    def forward(self, x, edge_index, type_masks):
        """
        x: [B, N, 7]
        edge_index: [2, E]              Topology is the same for all instance
        type_masks: dict with boolean masks:
          - type_masks['prod']  -> [N] True for generator nodes
          - type_masks['line']  -> [N] True for line nodes
          - type_masks['loc']   -> [N] True for location nodes

        THink about how to compute the unmet demand nodes based on predicted p and f.
        """

        h = self.encoder(x)
        h = torch.relu(h)

        for conv in self.convs:
            h = self.layer_norm(h)
            h = conv(h, edge_index)
            h = torch.relu(h)

        # Initialize outputs with zeros
        N = x.size(0)
        p = torch.zeros(N, 1, device=x.device)
        f = torch.zeros(N, 1, device=x.device)
        

        # Apply heads only on appropriate node types
        prod_mask = type_masks['prod']
        line_mask = type_masks['line']
        loc_mask  = type_masks['loc']   

        p[prod_mask]  = self.prod_head(h[prod_mask])
        f[line_mask]  = self.flow_head(h[line_mask])

        prod_out = self.prod_head(h[prod_mask])
        flow_out = self.flow_head(h[line_mask])

        B = int(x.size(0) / self.N)
        demand = x[type_masks['loc']][:,0]
        
        demand = demand.view(B, self.L) 
        
        p = prod_out.view(B, self.G)
        f = flow_out.view(B, self.F)
        

        md = self.compute_unmet_demand(p, f, demand)
        return p, f, md


    

class DualGNN(nn.Module):
    '''
    Graph-based Dual Network with KKT Reconstruction (Classification approach).

    1. GNN layers process the graph to extract features (h).
    2. A classification head predicts raw logits for lambda (nodal price) on Location nodes.
    3. Classification logic determines the final lambda value (lambda).
    4. The KKT Reconstruction layer (complete_duals) infers the inequality duals (mu)
        based on the predicted lambda and problem cost coefficients.
    '''
    def __init__(self, args, data, in_dim=7, hidden_size_factor=5.0, num_layers=3):
        super().__init__()
        self.data = data
        self.args = args
        
        # --- Device and Type Setup ---
        if args["device"] == "mps":
            self.DTYPE = torch.float32
            self.DEVICE = torch.device("mps")
        else:
            self.DTYPE = torch.float64
            self.DEVICE = torch.device("cpu")

        # --- Classification Setup ---
        # Set of possible dual classes (Negative costs: [-c_g, ..., -VOLL])
        self.classes = -1 * torch.concat([self.data.cost_vec.unique(), torch.tensor([self.data.pVOLL], dtype=self.DTYPE)])
        self.classes = self.classes.to(self.DEVICE)
        self.n_classes = self.classes.numel()
        self.n_dual_vars = data.neq  # Number of equality constraints (i.e., Location nodes |N|)

        # --- GNN Architecture Setup ---
        hidden_dim = int(hidden_size_factor * data.xdim) if args.get("hidden_size_factor") else 64
        
        self.encoder = torch.nn.Linear(in_dim, hidden_dim)

        self.convs = torch.nn.ModuleList([
            GraphConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # The head outputs a raw score for each class for lambda (applied only to Loc nodes)
        self.lambda_classifier_head = torch.nn.Linear(hidden_dim, self.n_classes)
        
        # --- KKT Reconstruction Tensors ---
        # These are constant tensors used in complete_duals
        self.eq_cm = data.eq_cm.to(self.DTYPE).to(self.DEVICE)
        self.obj_coeff = data.obj_coeff.to(self.DTYPE).to(self.DEVICE)
        self.num_g = data.num_g
        self.num_l = data.num_l
        
    def complete_duals(self, lamb):
        """
        KKT Reconstruction: Infer mu (inequality duals) from lambda (equality duals)
        using the stationarity condition, ensuring the dual solution is valid.
        """
        eq_cm_D_nt = self.eq_cm
        lamb_D_nt = lamb
        obj_coeff = self.obj_coeff

        # Calculate the raw mu vector: mu = ObjCoeff + A^T * lambda
        mu = obj_coeff + torch.matmul(lamb_D_nt, eq_cm_D_nt)

        # Compute lower and upper bound multipliers using ReLU (satisfies complementary slackness)
        mu_lb = torch.relu(mu)   # Lower bound multipliers (|mu|^+)
        mu_ub = torch.relu(-mu)  # Upper bound multipliers (|mu|^-)

        # Split and concatenate back into the original mu vector structure [p_lb, p_ub, f_lb, f_ub, md_lb, md_ub]
        
        # 1. Generator bounds (p_g)
        p_g_lb = mu_lb[:, :self.num_g]
        p_g_ub = mu_ub[:, :self.num_g]

        # 2. Line flow bounds (f_l)
        f_l_lb = mu_lb[:, self.num_g:self.num_g + self.num_l]
        f_l_ub = mu_ub[:, self.num_g:self.num_g + self.num_l]

        # 3. Unmet demand bounds (md_n)
        md_n_lb = mu_lb[:, self.num_g + self.num_l:]
        md_n_ub = mu_ub[:, self.num_g + self.num_l:]

        # Concatenate in the exact required order
        out_mu = torch.cat([p_g_lb, p_g_ub, f_l_lb, f_l_ub, md_n_lb, md_n_ub], dim=1)

        return out_mu
        
        
    def forward(self, batch):
        """
        Performs GNN propagation, lambda classification, and mu reconstruction.
        Node Feature:
        [Demand, MaxProd, FlowUB, FlowLB, LocNode, ProdNode, LineNode]
        batch: PyG Batch object containing x, edge_index, and type_masks.
        """
        
        # 1. GNN Propagation
        h = self.encoder(batch.x)
        h = torch.relu(h)
        
        for conv in self.convs:
            h = self.layer_norm(h)
            h = conv(h, batch.edge_index)
            h = torch.relu(h)

        # 2. Extract features for Location nodes and predict logits
        loc_mask = batch.type_masks['loc']
        h_loc  = h[loc_mask]
        lamb_logits = self.lambda_classifier_head(h_loc) # [N_loc_in_batch, n_classes]

        # 3. Classification Logic to get Lambda
        B = batch.num_graphs
        
        # Reshape logits to [Batch_size, n_var, n_classes]
        out_lamb_raw_probas = lamb_logits.view(B, self.n_dual_vars, self.n_classes)

        if self.training:
            # During training (differentiable), use softmax and expected value
            out_lamb_probas = torch.softmax(out_lamb_raw_probas, dim=-1)
            out_lamb = torch.sum(out_lamb_probas * self.classes, dim=-1) # [B, n_var]
        else:
            # During evaluation, use the max-likelihood class (sharp prediction)
            predicted_class = out_lamb_raw_probas.argmax(dim=-1)
            out_lamb = self.classes[predicted_class] # [B, n_var]
            
        # 4. KKT Reconstruction of Mu
        out_mu = self.complete_duals(out_lamb)
        
        # Returns [B, nineq] and [B, neq]
        return out_mu, out_lamb