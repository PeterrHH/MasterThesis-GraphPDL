import torch
import torch.nn as nn


class PrimalNet(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.data = data
        if args["hidden_size"]:
            self.hidden_sizes = [int(args["hidden_size"])] * args["n_layers"]
        else:
            self.hidden_sizes = [int(args["hidden_size_factor"]*data.xdim)] * args["n_layers"]
        
        # Create the list of layer sizes
        layer_sizes = [data.xdim] + self.hidden_sizes + [data.ydim]
        layers = []

        # layers.append(nn.LayerNorm(data.xdim))

        # Create layers dynamically based on the provided hidden_sizes
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            if out_size != data.ydim:  # Add ReLU activation for hidden layers only
                layers.append(nn.ReLU())

        # Initialize all layers
        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                # nn.init.xavier_uniform_(layer.weight)

        self.net = nn.Sequential(*layers)
    
    def forward(self, x, total_demands=None):
        #! If we are training, we do not scale the output by the total demands. Only for logging metrics.
        if not self.training and (total_demands != None):
            return self.net(x) * total_demands
        else:
            return self.net(x)

class DualNet(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.data = data
        if args["hidden_size"]:
            self.hidden_sizes = [int(args["hidden_size"])] * args["n_layers"]
        else:
            self.hidden_sizes = [int(args["hidden_size_factor"]*data.xdim)] * args["n_layers"]
        self.mu_size = self.data.nineq
        self.lamb_size = self.data.neq

        if args["device"] == "mps":
            self.DTYPE = torch.float32
            self.DEVICE = torch.device("mps")
        else:
            self.DTYPE = torch.float64
            self.DEVICE = torch.device("cpu")

        torch.set_default_dtype(self.DTYPE)
        torch.set_default_device(self.DEVICE)

        # Create the list of layer sizes
        layer_sizes = [data.xdim] + self.hidden_sizes
        # layer_sizes = [2*data.xdim + 1000] + self.hidden_sizes
        layers = []
        # Create layers dynamically based on the provided hidden_sizes
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())

        # Initialize all layers
        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)

        # Add the output layer
        self.out_layer = nn.Linear(self.hidden_sizes[-1], self.mu_size + self.lamb_size)
        nn.init.zeros_(self.out_layer.weight)  # Initialize output layer weights to 0
        nn.init.zeros_(self.out_layer.bias)    # Initialize output layer biases to 0
        layers.append(self.out_layer)

        self.net = nn.Sequential(*layers)
    
    def forward(self, x, *args):
        out = self.net(x)
        out_mu = out[:, :self.mu_size]
        out_lamb = out[:, self.mu_size:]
        return out_mu, out_lamb


class DualNetTwoOutputLayers(nn.Module):
    def __init__(self, data, hidden_size):
        super().__init__()
        self.data = data
        self.hidden_size = hidden_size
        layer_sizes = [data.xdim, self.hidden_size, self.hidden_size]
        layers = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
        for layer in layers:
            if type(layer) == nn.Linear:
                nn.init.kaiming_normal_(layer.weight)
        self.out_layer_mu = nn.Linear(self.hidden_size, data.nineq)
        self.out_layer_lamb = nn.Linear(self.hidden_size, data.neq)
        # Init last layers as 0, like in the paper
        nn.init.zeros_(self.out_layer_mu.weight)
        nn.init.zeros_(self.out_layer_mu.bias)
        nn.init.zeros_(self.out_layer_lamb.weight)
        nn.init.zeros_(self.out_layer_lamb.bias)

        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.net(x)
        out_mu = self.out_layer_mu(out)
        out_lamb = self.out_layer_lamb(out)
        return out_mu, out_lamb

class FeedForwardNet(nn.Module):
    def __init__(self, args, input_dim, hidden_sizes, output_dim, layernorm=None):
        """_summary_

        Args:
            input_dim (int): Number of input features
            output_dim (int): Number of output features
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.output_dim = output_dim

        if args["device"] == "mps":
            self.DTYPE = torch.float32
            self.DEVICE = torch.device("mps")
        else:
            self.DTYPE = torch.float64
            self.DEVICE = torch.device("cpu")

        torch.set_default_dtype(self.DTYPE)
        torch.set_default_device(self.DEVICE)

        # Create the list of layer sizes
        layer_sizes = [self.input_dim] + self.hidden_sizes + [self.output_dim]
        layers = []
        
        if layernorm is None:
            if args["layernorm"]:
                layers.append(nn.LayerNorm(input_dim)) #! This is necessary to prevent gradient saturation in the sigmoids.
        elif layernorm is True:
            layers.append(nn.LayerNorm(input_dim))

        # Create layers dynamically based on the provided hidden_sizes
        for idx, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layers.append(nn.Linear(in_size, out_size))
            # Add ReLU only if it is not the last layer
            if idx < len(layer_sizes) - 2:  # The last layer does not need ReLU
                layers.append(nn.ReLU())
        
        # Initialize all layers
        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                # nn.init.xavier_uniform_(layer.weight)
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        x_out = self.net(x)
        return x_out


class BoundRepairLayer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, lb, ub, k=1.0):
        """_summary_

        Args:
            x (_type_): Decision variables, shape [B, N, T]
            lb (_type_): Lower bounds of decision variables, shape [B, N, T]
            ub (_type_): Upper bounds of decision variables, shape [B, N, T]

        Returns:
            _type_: _description_
        """
        scaled = torch.sigmoid(k * x)

        repaired = lb + (ub - lb) * scaled

        if False: #! Set to True to attach hooks to inspect gradients
            def print_grad(name):
                return lambda grad: print(f"Gradient for {name}: {grad.norm():.4e}")
            x.register_hook(print_grad("x"))
            scaled.register_hook(print_grad("sigmoid(kx)"))
            repaired.register_hook(print_grad("repaired output"))

        return repaired
class EstimateSlackLayer(nn.Module):
    def __init__(self, node_to_gen_mask, lineflow_mask):
        super().__init__()

        self.node_to_gen_mask = node_to_gen_mask    # [N, G]
        self.lineflow_mask = lineflow_mask          # [N, L]

    def forward(self, p_gt, f_lt, D_nt):
        """Compute md_n,t

        Args:
            p_gt (_type_): Generator production, shape [B, G]
            f_lt (_type_): Line flow, shape [B, L]
            D_nt (_type_): Demand, shape [B, N]
        """
        combined_flow = torch.matmul(p_gt, self.node_to_gen_mask.T) + \
                        torch.matmul(f_lt, self.lineflow_mask.T)
    
        md_nt = D_nt - combined_flow

        return md_nt

class PrimalNetEndToEnd(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.data = data
        self.hidden_sizes = [int(args["hidden_size_factor"]*data.xdim)] * args["n_layers"]
        self.args = args

        if not self.args["repair"]:
            assert not self.args["repair_bounds"], "If repair is disabled, bounds repair should not be enabled."
            assert not self.args["repair_completion"], "If repair is disabled, completion repair should not be enabled."
            assert not self.args["repair_power_balance"], "If repair is disabled, power balance repair should not be enabled."

        if self.args["repair_power_balance"]:
            # Power balance repair requires bounds repair.
            assert self.args["repair_bounds"], "Power balance repair requires bounds repair."
        
        if self.args["repair_bounds"]:
            # Bound repairs need layernorm to prevent gradient saturation in sigmoids.
            assert self.args["layernorm"], "Bounds repair requires layernorm."

        if self.args["device"] == "mps":
                self.DTYPE = torch.float32
                self.DEVICE = torch.device("mps")
        else:
            self.DTYPE = torch.float64
            self.DEVICE = torch.device("cpu")

        torch.set_default_dtype(self.DTYPE)
        torch.set_default_device(self.DEVICE)

        # TODO: Implement compact benders form.
        # if self.data.args["benders_compact"]:
            # self.out_dim = data.num_g + data.n_prod_vars + data.n_line_vars
        # else:
        if args["repair_completion"]:
            self.out_dim = data.n_prod_vars + data.n_line_vars
        else:
            self.out_dim = data.n_prod_vars + data.n_line_vars + data.n_md_vars

        self.feed_forward = FeedForwardNet(args, data.xdim, self.hidden_sizes, output_dim=self.out_dim).to(self.DTYPE).to(self.DEVICE)
        
        if self.args["repair_bounds"]:
            self.bound_repair_layer = BoundRepairLayer()
            # self.ramping_repair_layer = RampingRepairLayer()

        if self.args["repair_completion"]:
            self.estimate_slack_layer = EstimateSlackLayer(data.node_to_gen_mask.to(self.DTYPE).to(self.DEVICE), data.lineflow_mask.to(self.DTYPE).to(self.DEVICE))
    
    def forward(self, x, total_demands=None):
        eq_rhs, ineq_rhs = self.data.split_X(x)
        x_out = self.feed_forward(x)
        if not self.args["repair"]:
            return x_out
        # [B, G, T], [B, L, T]
        if self.args["repair_completion"]:
            ui_g, p_gt, f_lt = self.data.split_dec_vars_from_Y_raw(x_out)
        else:
            p_gt, f_lt, md_nt = self.data.split_dec_vars_from_Y(x_out)

        # [B, bounds, T]
        p_gt_lb, p_gt_ub, f_lt_lb, f_lt_ub, md_nt_lb, md_nt_ub = self.data.split_ineq_constraints(ineq_rhs)
        
        if self.args["repair_bounds"]:
            p_gt = self.bound_repair_layer(p_gt, p_gt_lb, p_gt_ub)
                # Lineflow lower bound is negative.
                #! Note: Lineflows cannot be repaired more, since they depend on other lineflows. 
                #! For example, if we repair a lineflow such that it cannot export more than (imports + generation),
                #! then it will affect other lineflows, and we would need to repair them again.
            f_lt = self.bound_repair_layer(f_lt, -f_lt_lb, f_lt_ub) #! Bounds need to be repaired for this to work!

        if self.args["repair_power_balance"]:
            net_flow = torch.matmul(f_lt, self.data.lineflow_mask.T)  # [B, N]
            updated_demand = eq_rhs - net_flow                                    # [B, N]
            # Clamp demand between lower and upper bound of total capacity, since generators must produce between that.
            total_capacity_lb = torch.matmul(p_gt_lb, self.data.node_to_gen_mask.T)
            total_capacity_ub = torch.matmul(p_gt_ub, self.data.node_to_gen_mask.T)
            demand_clamped = torch.clamp(updated_demand, min=total_capacity_lb, max=total_capacity_ub)

            # Repair generation.
            total_generation = torch.matmul(p_gt, self.data.node_to_gen_mask.T)
            # Calculate zeta_up and zeta_down using vectorized operations
            mask_up = (total_generation < demand_clamped).to(self.DTYPE)
            # Calculate zeta_up and zeta_down based on conditions
            zeta_up = (demand_clamped - total_generation) / ((total_capacity_ub - total_generation) + 1e-12)
            zeta_down = (total_generation - demand_clamped) / ((total_generation - total_capacity_lb) + 1e-12)

            # Expand masks to generator dimension.
            mask_up = torch.matmul(mask_up, self.data.node_to_gen_mask)
            zeta_up = torch.matmul(zeta_up, self.data.node_to_gen_mask)
            zeta_down = torch.matmul(zeta_down, self.data.node_to_gen_mask)
            # Apply the updates to p_gt based on the condition
            p_gt_repaired = torch.where(mask_up.bool(), (1 - zeta_up) * p_gt + zeta_up * p_gt_ub, (1 - zeta_down) * p_gt + zeta_down * p_gt_lb)
            # p_gt = torch.where(mask_down, (1 - zeta_down) * p_gt + zeta_down * p_gt_lb, p_gt)

            p_gt = p_gt_repaired

        if self.args["repair_completion"]:
            UI_g, D_nt = self.data.split_eq_constraints(eq_rhs)
            md_nt = self.estimate_slack_layer(p_gt, f_lt, D_nt)

        y = torch.cat([p_gt, f_lt, md_nt], dim=1)
        # y = torch.cat([md_nt, f_lt, p_gt], dim=1)

        # Only scale if we are not training.
        if not self.training and (total_demands != None):
            return y * total_demands
        else:
            return y

class DualNetEndToEnd(nn.Module):
    def __init__(self, args, data, hidden_size_factor=5.0, n_layers=4):
        super().__init__()
        self.data = data
        self.hidden_sizes = [int(hidden_size_factor*data.xdim)] * n_layers
        self.args = args
        self.ED_args = args["ED_args"]

        if self.ED_args["benders_compact"]:
            self.out_dim = data.num_g + data.neq
        else:
            self.out_dim = data.neq

        if args["device"] == "mps":
            self.DTYPE = torch.float32
            self.DEVICE = torch.device("mps")
        else:
            self.DTYPE = torch.float64
            self.DEVICE = torch.device("cpu")

        #! Only predict lambda, we infer mu from it.
        self.feed_forward = FeedForwardNet(args, data.xdim, self.hidden_sizes, output_dim=self.out_dim, layernorm=True).to(self.DTYPE).to(self.DEVICE)
    
    def complete_duals(self, lamb):
        eq_cm_D_nt = self.data.eq_cm
        lamb_D_nt = lamb
        obj_coeff = self.data.obj_coeff

        mu = obj_coeff + torch.matmul(lamb_D_nt, eq_cm_D_nt)

        # Compute lower and upper bound multipliers
        mu_lb = torch.relu(mu)   # Lower bound multipliers |mu|^+
        mu_ub = torch.relu(-mu)  # Upper bound multipliers |mu|^-

        # Split into groups, following the exact structure of mu
        p_g_lb = mu_lb[:, :self.data.num_g]  # Lower bounds for p_g
        p_g_ub = mu_ub[:, :self.data.num_g]  # Upper bounds for p_g

        f_l_lb = mu_lb[:, self.data.num_g:self.data.num_g + self.data.num_l]  # Lower bounds for f_l
        f_l_ub = mu_ub[:, self.data.num_g:self.data.num_g + self.data.num_l]  # Upper bounds for f_l

        md_n_lb = mu_lb[:, self.data.num_g + self.data.num_l:]  # Lower bounds for md_n
        md_n_ub = mu_ub[:, self.data.num_g + self.data.num_l:]  # Upper bounds for md_n

        # Concatenate while maintaining order
        out_mu = torch.cat([
            p_g_lb, p_g_ub,  # Lower and Upper bounds for p_g
            f_l_lb, f_l_ub,  # Lower and Upper bounds for f_l
            md_n_lb, md_n_ub  # Lower and Upper bounds for md_n
        ], dim=1)

        return out_mu
        
        
    def forward(self, x):
        out_lamb = self.feed_forward(x)
        out_mu = self.complete_duals(out_lamb)
        # print(out_lamb)
        return out_mu, out_lamb
    
class DualClassificationNetEndToEnd(nn.Module):
    def __init__(self, args, data, hidden_size_factor=5.0, n_layers=4):
        super().__init__()
        self.data = data
        self.hidden_sizes = [int(hidden_size_factor*data.xdim)] * n_layers
        self.args = args
        self.ED_args = args["ED_args"]

        # Objective coefficients contain all costs for all generators and unmet demand.
        self.classes = -1 * torch.concat([self.data.cost_vec.unique(), torch.tensor([self.data.pVOLL])])
        self.n_classes = self.classes.numel()
        self.n_dual_vars = data.neq

        #! For each dual variable, We now predict probabilities for each class
        self.out_dim = self.n_classes * self.n_dual_vars

        if args["device"] == "mps":
            self.DTYPE = torch.float32
            self.DEVICE = torch.device("mps")
        else:
            self.DTYPE = torch.float64
            self.DEVICE = torch.device("cpu")

        #! Only predict lambda, we infer mu from it.
        #! Softmax requires layer norm.
        self.feed_forward = FeedForwardNet(args, data.xdim, self.hidden_sizes, output_dim=self.out_dim, layernorm=True).to(self.DTYPE).to(self.DEVICE)
    
    def complete_duals(self, lamb):
        eq_cm_D_nt = self.data.eq_cm
        lamb_D_nt = lamb
        obj_coeff = self.data.obj_coeff

        mu = obj_coeff + torch.matmul(lamb_D_nt, eq_cm_D_nt)

        # Compute lower and upper bound multipliers
        mu_lb = torch.relu(mu)   # Lower bound multipliers |mu|^+
        mu_ub = torch.relu(-mu)  # Upper bound multipliers |mu|^-

        # Split into groups, following the exact structure of mu
        p_g_lb = mu_lb[:, :self.data.num_g]  # Lower bounds for p_g
        p_g_ub = mu_ub[:, :self.data.num_g]  # Upper bounds for p_g

        f_l_lb = mu_lb[:, self.data.num_g:self.data.num_g + self.data.num_l]  # Lower bounds for f_l
        f_l_ub = mu_ub[:, self.data.num_g:self.data.num_g + self.data.num_l]  # Upper bounds for f_l

        md_n_lb = mu_lb[:, self.data.num_g + self.data.num_l:]  # Lower bounds for md_n
        md_n_ub = mu_ub[:, self.data.num_g + self.data.num_l:]  # Upper bounds for md_n

        # Concatenate while maintaining order
        out_mu = torch.cat([
            p_g_lb, p_g_ub,  # Lower and Upper bounds for p_g
            f_l_lb, f_l_ub,  # Lower and Upper bounds for f_l
            md_n_lb, md_n_ub  # Lower and Upper bounds for md_n
        ], dim=1)

        return out_mu
        
        
    def forward(self, x):
        out_lamb_raw_probas = self.feed_forward(x) # [B, n_var*n_classes]
        T = 1
        out_lamb_raw_probas = out_lamb_raw_probas.view(-1, self.n_dual_vars, self.n_classes) # [B, n_var, n_classes]

        if self.training:
            out_lamb_probas = torch.softmax(out_lamb_raw_probas, dim=-1)

            out_lamb = torch.sum(out_lamb_probas * self.classes, dim=-1)
            out_mu = self.complete_duals(out_lamb)

            return out_mu, out_lamb

        else:
            predicted_class = out_lamb_raw_probas.argmax(dim=-1)
            out_lamb = self.classes[predicted_class]
            out_mu = self.complete_duals(out_lamb)
            return out_mu, out_lamb
         
def load(args, data, save_dir):
    primal_net = PrimalNetEndToEnd(args, data=data)
    primal_net.load_state_dict(torch.load(save_dir + '/primal_weights.pth', weights_only=True))
    dual_net = DualNetEndToEnd(args, data=data)
    dual_net.load_state_dict(torch.load(save_dir + '/dual_weights.pth', weights_only=True))

    return primal_net, dual_net