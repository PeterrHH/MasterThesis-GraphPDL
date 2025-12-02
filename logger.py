import torch
from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger():
    def __init__(self, args, data, X, scale_factors, train_indices, valid_indices, save_dir, opt_targets):
        self.writer = SummaryWriter(log_dir=save_dir)
        self.opt_targets = opt_targets
        self.args = args
        self.data = data
        self.problem_type = args["problem_type"]
        self.is_qp = self.problem_type == "QP"
        self.is_qp_simple = self.problem_type == "QP" and args["QP_args"]["type"] == "simple"
        self.is_qp_not_simple = self.problem_type == "QP" and args["QP_args"]["type"] != "simple"

        if args["device"] == "mps":
            self.DTYPE = torch.float32
            self.DEVICE = torch.device("mps")
        else:
            self.DTYPE = torch.float64
            self.DEVICE = torch.device("cpu")

        self.X_train = X[train_indices].to(self.DTYPE).to(self.DEVICE)
        self.X_valid = X[valid_indices].to(self.DTYPE).to(self.DEVICE)
        self.scale_factors_train = scale_factors[train_indices]
        self.scale_factors_valid = scale_factors[valid_indices]

        self.train_indices = train_indices
        self.valid_indices = valid_indices

        if self.opt_targets:
            if not self.is_qp:
                self.Y_target_train = data.opt_targets["y_operational"].to(self.DTYPE).to(self.DEVICE)[self.train_indices]
                self.mu_target_train = data.opt_targets["mu_operational"].to(self.DTYPE).to(self.DEVICE)[self.train_indices]  
                self.lamb_target_train = data.opt_targets["lamb_operational"].to(self.DTYPE).to(self.DEVICE)[self.train_indices]
                self.Y_target_valid = data.opt_targets["y_operational"].to(self.DTYPE).to(self.DEVICE)[self.valid_indices]
                self.mu_target_valid = data.opt_targets["mu_operational"].to(self.DTYPE).to(self.DEVICE)[self.valid_indices]  
                self.lamb_target_valid = data.opt_targets["lamb_operational"].to(self.DTYPE).to(self.DEVICE)[self.valid_indices]
            elif self.is_qp_simple:
                self.Y_target_train = data.trainY
                self.mu_target_train = data.train_mu
                self.lamb_target_train = data.train_lamb
                self.Y_target_valid = data.validY
                self.mu_target_valid = data.valid_mu
                self.lamb_target_valid = data.valid_lamb
            elif self.is_qp_not_simple:
                self.Y_target_train = data.trainY
                self.Y_target_valid = data.validY

    def log_primal_loss(self, loss, obj, lagrange_eq, lagrange_ineq, penalty, step):

        self.writer.add_scalar(f"Train_loss/primal_loss", loss, step)
        self.writer.add_scalar(f"Train_loss_components/obj", obj, step)
        self.writer.add_scalar(f"Train_loss_components/primal_lagrange_eq", lagrange_eq, step)
        self.writer.add_scalar(f"Train_loss_components/primal_lagrange_ineq", lagrange_ineq, step)
        self.writer.add_scalar(f"Train_loss_components/primal_penalty", penalty, step)
    
    def log_dual_loss(self, loss, step, obj, lagrange_eq, lagrange_ineq, penalty):
        self.writer.add_scalar(f"Train_loss/dual_loss", loss, step)
        self.writer.add_scalar(f"dual_loss_components/obj", obj, step)
        self.writer.add_scalar(f"dual_loss_components/lagrange_eq", lagrange_eq, step)
        self.writer.add_scalar(f"dual_loss_components/lagrange_ineq", lagrange_ineq, step)
        self.writer.add_scalar(f"dual_loss_components/dual_penalty", penalty, step)



    def log_train(self, data, primal_net, dual_net, rho, step):
        with torch.no_grad():
            primal_net.eval()
            dual_net.eval()
            Y = primal_net(self.X_train, self.scale_factors_train)
            mu, lamb = dual_net(self.X_train)

            obj = data.obj_fn(self.X_train, Y)

            ineq_resid = data.ineq_resid(self.X_train, Y)
            ineq_dist = data.ineq_dist(self.X_train, Y)

            eq_resid = data.eq_resid(self.X_train, Y)

            if self.opt_targets:
                obj_target = data.obj_fn(self.X_train, self.Y_target_train)
                if not self.is_qp_not_simple:
                    dual_obj = data.dual_obj_fn(self.X_train, mu, lamb)
                    dual_obj_target = data.dual_obj_fn(self.X_train, self.mu_target_train, self.lamb_target_train)
                    self.writer.add_scalar(f"Train_obj/dual_obj_optimality_gap", ((dual_obj_target - dual_obj)/dual_obj_target).mean(), step)
                    self.writer.add_scalar(f"Train_obj/dual_obj", dual_obj.mean(), step)
                    self.writer.add_scalar(f"Train_obj/duality_gap", ((obj - dual_obj)/obj).mean(), step)

                optimality_gap = (obj - obj_target)/obj_target
                self.writer.add_scalar(f"Train_obj/obj_optimality_gap", optimality_gap.mean(), step)
            
            # Obj funcs
            self.writer.add_scalar(f"Train_obj/obj", obj.mean(), step)
            

            #! Neural network outputs and targets
            # self.writer.add_scalar(f"Train_outputs/Y", Y.mean(), step)
            # self.writer.add_scalar(f"Train_outputs/mu", mu.mean(), step)
            # self.writer.add_scalar(f"Train_outputs/lamb", lamb.mean(), step)
            # if self.opt_targets:
            #     if data.args["benders_compact"]:
            #         Y_diff = (Y[:, data.num_g:] - self.Y_target_train).abs()
            #         lamb_diff = (lamb[:, data.num_g:] - self.lamb_target_train).abs()
            #     else:
            #         Y_diff = (Y - self.Y_target_train).abs()
            #         lamb_diff = (lamb - self.lamb_target_train).abs()
            #     mu_diff = (mu - self.mu_target_train).abs()
            #     self.writer.add_scalar(f"Train_outputs/Y_diff", Y_diff.mean(), step)
            #     self.writer.add_scalar(f"Train_outputs/mu_diff", mu_diff.mean(), step)
            #     self.writer.add_scalar(f"Train_outputs/lamb_diff", lamb_diff.mean(), step)

            #! Constraint violations
            # self.writer.add_scalar(f"Train_constraints/eq_resid", eq_resid.mean(), step)
            # self.writer.add_scalar(f"Train_constraints/ineq_resid", ineq_resid.mean(), step)
            # self.writer.add_scalar(f"Train_constraints/ineq_mean", ineq_dist.mean(), step)
            # self.writer.add_scalar(f"Train_constraints/ineq_max", ineq_dist.max(), step)
            # self.writer.add_scalar(f"Train_constraints/eq_mean", eq_resid.abs().mean(), step)
            # self.writer.add_scalar(f"Train_constraints/eq_max", eq_resid.abs().max(), step)

            if not self.is_qp:
                p_gt, f_lt, md_nt = data.split_dec_vars_from_Y(Y)
                eq_rhs_train, ineq_rhs_train = data.split_X(self.X_train)
                p_gt_lb, p_gt_ub, f_lt_lb, f_lt_ub, md_nt_lb, md_nt_ub = data.split_ineq_constraints(ineq_rhs_train)
                p_gt_ub = p_gt_ub * self.scale_factors_train
                for i in range(p_gt.shape[1]):
                    self.writer.add_scalar(f"Train_decvars/generator{i}", p_gt[:, i].mean(), step)
                    #! First g of ineq_rhs are lower bounds, second g are upper bounds
                    self.writer.add_scalar(f"Train_decvars/generator{i}_frac", (p_gt[:, i]/p_gt_ub[:, i]).mean(), step)
                for i in range(f_lt.shape[1]):
                    self.writer.add_scalar(f"Train_decvars/lineflow{i}", f_lt[:, i].mean(), step)
                for i in range(md_nt.shape[1]):
                    self.writer.add_scalar(f"Train_decvars/missed_demand{i}", md_nt[:, i].mean(), step)
                for i in range(md_nt.shape[1]):
                    self.writer.add_scalar(f"Train_decvars/missed_demand_abs{i}", md_nt[:, i].abs().mean(), step)

            if not self.is_qp:
                if self.opt_targets:
                    # Primal variable specific differences
                    
                    p_gt_target, f_lt_target, md_nt_target = data.split_dec_vars_from_Y(self.Y_target_train, log=True)
                    diff_p_gt = p_gt - p_gt_target
                    diff_f_lt = f_lt - f_lt_target
                    diff_md_nt = md_nt - md_nt_target

                    net_flow = data.net_flow(f_lt)
                    net_flow_target = data.net_flow(f_lt_target)
                    diff_net_flow = net_flow - net_flow_target

                    # diff_ui_g = (Y[:, data.ui_g_indices] - Y_target[:, data.ui_g_indices])
                    self.writer.add_scalar(f"Train_var_diffs/diff_p_gt", diff_p_gt.mean(), step)
                    self.writer.add_scalar(f"Train_var_diffs/diff_f_lt", diff_f_lt.mean(), step)
                    self.writer.add_scalar(f"Train_var_diffs/diff_md_nt", diff_md_nt.mean(), step)
                    self.writer.add_scalar(f"Train_var_diffs/diff_net_flow", diff_net_flow.mean(), step)
                    # self.writer.add_scalar(f"Train_var_diffs/diff_ui_g", diff_ui_g.mean(), step)

                h, b, d, e, i, j = data.split_ineq_constraints(ineq_dist)
                ui_g, c = data.split_eq_constraints(eq_resid)

                self.writer.add_scalar(f"Train_constraint_specific/p_gt_ub", b.mean(), step)
                self.writer.add_scalar(f"Train_constraint_specific/node_balance", c.abs().mean(), step)
                self.writer.add_scalar(f"Train_constraint_specific/f_lt_lb", d.mean(), step)
                self.writer.add_scalar(f"Train_constraint_specific/f_lt_ub", e.mean(), step)
                # self.writer.add_scalar(f"Train_constraint_specific/f", f.mean(), step)
                # self.writer.add_scalar(f"Train_constraint_specific/g", g.mean(), step)
                self.writer.add_scalar(f"Train_constraint_specific/p_gt_lb", h.mean(), step)
                self.writer.add_scalar(f"Train_constraint_specific/md_nt_lb", i.mean(), step)
                self.writer.add_scalar(f"Train_constraint_specific/md_nt_ub", j.mean(), step)
            
            # if self.opt_targets: 
            #     # Dual variable specific differences
            #     # inequality
            #     mu_h, mu_b, mu_d, mu_e, mu_i, mu_j = data.split_ineq_constraints(mu)
            #     mu_target_h, mu_target_b, mu_target_d, mu_target_e, mu_target_i, mu_target_j = data.split_ineq_constraints(self.mu_target_train)
            #     mu_h_diff = mu_target_h - mu_h
            #     mu_b_diff = mu_target_b - mu_b
            #     mu_d_diff = mu_target_d - mu_d
            #     mu_e_diff = mu_target_e - mu_e
            #     mu_i_diff = mu_target_i - mu_i
            #     mu_j_diff = mu_target_j - mu_j
            #     # # equality
            #     ui_g, lamb_c = data.split_eq_constraints(lamb)
            #     ui_g, lamb_target_c = data.split_eq_constraints(self.lamb_target_train, log=True)
            #     lamb_c_diff = lamb_target_c - lamb_c

            #     self.writer.add_scalar(f"Train_dual_var_diffs/gen_ub", mu_b_diff.mean(), step)
            #     self.writer.add_scalar(f"Train_dual_var_diffs/node_balance", lamb_c_diff.mean(), step)
            #     self.writer.add_scalar(f"Train_dual_var_diffs/lineflow_lb", mu_d_diff.mean(), step)
            #     self.writer.add_scalar(f"Train_dual_var_diffs/lineflow_ub", mu_e_diff.mean(), step)
            #     self.writer.add_scalar(f"Train_dual_var_diffs/gen_lb", mu_h_diff.mean(), step)
            #     self.writer.add_scalar(f"Train_dual_var_diffs/md_lb", mu_i_diff.mean(), step)
            #     self.writer.add_scalar(f"Train_dual_var_diffs/md_ub", mu_j_diff.mean(), step)

            # Dual constraints
            if not self.is_qp_not_simple:
                dual_eq_resid = data.dual_eq_resid(mu, lamb)
                dual_ineq_resid = data.dual_ineq_resid(mu, lamb)
                dual_ineq_dist = torch.clamp(dual_ineq_resid, 0)
                self.writer.add_scalar("Dual_constraints/eq_resid", dual_eq_resid.abs().mean(), step)
                self.writer.add_scalar("Dual_constraints/ineq_mean", dual_ineq_dist.mean(), step)


            # Log gradients
            # Iterate over all layers and log their gradients
            for name, param in primal_net.named_parameters():
                if param.grad is not None:  # Skip parameters without gradients
                    self.writer.add_scalar(f"Gradients_primal/{name}", param.grad.norm().item(), step)
            
            for name, param in dual_net.named_parameters():
                if param.grad is not None:  # Skip parameters without gradients
                    self.writer.add_scalar(f"Gradients_dual/{name}", param.grad.norm().item(), step)

    def log_rho_vk(self, rho, v_k, step):
        self.writer.add_scalar(f"Rho_and_violation/rho", rho, step)
        self.writer.add_scalar(f"Rho_and_violation/v_k", v_k, step)

    def log_val(self, data, primal_net, dual_net, step):
        with torch.no_grad():
            primal_net.eval()
            dual_net.eval()
            Y = primal_net(self.X_valid, self.scale_factors_valid)
            mu, lamb = dual_net(self.X_valid)
            obj = data.obj_fn(self.X_valid, Y) # Containes penalization of negative missed demand
            

            if self.opt_targets:
                # Y_target = data.opt_targets["y_operational"][data.valid_indices]
                # mu_target = data.opt_targets["mu_operational"][data.valid_indices]
                # lamb_target = data.opt_targets["lamb_operational"][data.valid_indices]
                obj_target = data.obj_fn(self.X_valid, self.Y_target_valid)
                self.writer.add_scalar(f"Validation/obj_optimality_gap", ((obj - obj_target)/obj_target).mean(), step)
                if not self.is_qp_not_simple:
                    dual_obj = data.dual_obj_fn(self.X_valid, mu, lamb)
                    dual_obj_target = data.dual_obj_fn(self.X_valid, self.mu_target_valid, self.lamb_target_valid)
                    
                    self.writer.add_scalar(f"Validation/dual_obj_optimality_gap", ((dual_obj_target - dual_obj)/dual_obj_target).mean(), step)
                    self.writer.add_scalar(f"Validation/dual_obj", dual_obj.mean(), step)

            ineq_dist = data.ineq_dist(self.X_valid, Y)

            eq_resid = data.eq_resid(self.X_valid, Y)

            # Obj funcs
            self.writer.add_scalar(f"Validation/obj", obj.mean(), step)
            # Constraint violations
            self.writer.add_scalar(f"Validation/ineq_mean", ineq_dist.mean(), step)
            self.writer.add_scalar(f"Validation/ineq_max", ineq_dist.max(dim=1)[0].mean(), step)
            self.writer.add_scalar(f"Validation/eq_mean", eq_resid.abs().mean(), step)
            self.writer.add_scalar(f"Validation/eq_max", eq_resid.abs().max(dim=1)[0].mean(), step)