import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import wandb
from PIL import Image
import io
import pickle
import time

from gep_config_parser import *
from primal_dual import PrimalDualTrainer, load
from create_gep_dataset import prep_data

from torchviz import make_dot
from IPython.display import display

def plot_loss_landscape(trainer, primal_net, dual_net, loss_fn, dataloader, num_points=20, alpha=1.0, k=0):
    # Store original parameters
    original_params = [p.clone() for p in primal_net.parameters()]
    
    # Calculate two random directions
    direction1 = [torch.randn_like(p) for p in primal_net.parameters()]
    direction2 = [torch.randn_like(p) for p in primal_net.parameters()]
    
    # Normalize directions
    norm1 = torch.sqrt(sum(torch.sum(d**2) for d in direction1))
    norm2 = torch.sqrt(sum(torch.sum(d**2) for d in direction2))
    direction1 = [d / norm1 for d in direction1]
    direction2 = [d / norm2 for d in direction2]
    
    # Create grid
    x = np.linspace(-alpha, alpha, num_points)
    y = np.linspace(-alpha, alpha, num_points)
    X, Y = np.meshgrid(x, y)
    
    # Calculate loss for each point
    Z = np.zeros_like(X)
    for i in range(num_points):
        for j in range(num_points):
            # Update model parameters
            for p, d1, d2 in zip(primal_net.parameters(), direction1, direction2):
                p.data = p.data + X[i,j] * d1 + Y[i,j] * d2
            
            # Calculate loss
            total_loss = 0
            num_batches = 0
            for Xtrain, sample_indices in dataloader:
                train_eq_cm, train_ineq_cm, train_eq_rhs, train_ineq_rhs = trainer.eq_cm_train[sample_indices], trainer.ineq_cm_train[sample_indices], trainer.eq_rhs_train[sample_indices], trainer.ineq_rhs_train[sample_indices]
                outputs = primal_net(Xtrain, train_eq_rhs, train_ineq_rhs)
                if k == 0:
                    mu, lamb = torch.zeros_like(train_ineq_rhs), torch.zeros_like(train_eq_rhs)
                else:
                    mu, lamb = dual_net(Xtrain, train_eq_cm)
                loss = loss_fn(trainer, outputs, train_eq_cm, train_ineq_cm, train_eq_rhs, train_ineq_rhs, mu, lamb).mean()
                total_loss += loss.item()
                num_batches += 1
            Z[i,j] = total_loss / num_batches
            
            # Reset model parameters
            for p, orig_p in zip(primal_net.parameters(), original_params):
                p.data = orig_p.clone()
    
    # Plot the loss landscape
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('Direction 1')
    ax.set_ylabel('Direction 2')
    ax.set_zlabel('Loss')
    ax.set_title('Loss Landscape')
    fig.colorbar(surf)
    
    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    plt.show()

    # Close the plot to free up memory
    # plt.close(fig)
    
    # return buf

def plot_loss_landscape_output(trainer, primal_net, dual_net, loss_fn, dataloader, num_points=75, alpha=1, k=0):
    """
    Plots the loss landscape as a function of the output variable.
    
    Instead of modifying the model parameters, we perturb the output (y)
    along two random directions. For each candidate output, we compute the loss.
    
    Args:
        primal_net: The network that produces outputs y.
        dual_net: Dual network to compute dual variables if needed.
        loss_fn: Function to compute loss, with signature:
                 loss_fn(y, train_eq_cm, train_ineq_cm, train_eq_rhs, train_ineq_rhs, mu, lamb)
        dataloader: A dataloader yielding one batch of data.
        num_points: Number of grid points along each direction.
        alpha: Maximum perturbation magnitude in each direction.
        k: If zero, use zeros for dual variables; otherwise use dual_net.
    """
    # Get one batch from the dataloader
    batch = next(iter(dataloader))
    Xtrain, sample_indices = batch
    
    # Compute the reference output from the network
    with torch.no_grad():
        original_output = primal_net(Xtrain, trainer.eq_rhs_train[sample_indices], trainer.ineq_rhs_train[sample_indices])
    
    # Generate two random directions in the output space (same shape as output)
    direction1 = torch.randn_like(original_output)
    direction2 = torch.randn_like(original_output)
    
    # Normalize the directions (using the overall norm)
    direction1 = direction1 / torch.norm(direction1)
    direction2 = direction2 / torch.norm(direction2)
    
    # Create a grid of perturbations (scalars)
    x_grid = np.linspace(-alpha, alpha, num_points)
    y_grid = np.linspace(-alpha, alpha, num_points)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Allocate grid for loss values
    Z = np.zeros_like(X)
    
    # For each grid point, perturb the output and compute loss.
    for i in range(num_points):
        for j in range(num_points):
            # Candidate output: original output plus a perturbation in output space
            candidate_output = original_output + X[i, j] * direction1 + Y[i, j] * direction2
            
            # Determine dual variables (if needed)
            if k == 0:
                mu = torch.zeros_like(trainer.ineq_rhs_train[sample_indices])
                lamb = torch.zeros_like(trainer.eq_rhs_train[sample_indices])
            else:
                mu, lamb = dual_net(Xtrain, trainer.eq_cm_train[sample_indices])
            
            # Compute loss using candidate output.
            # (Assume loss_fn is written to take the candidate output as its first argument.)
            loss_val = loss_fn(trainer, candidate_output, trainer.eq_cm_train[sample_indices], trainer.ineq_cm_train[sample_indices], trainer.eq_rhs_train[sample_indices], trainer.ineq_rhs_train[sample_indices], mu, lamb).mean()
            Z[i, j] = loss_val.item()
    
    # Plot the loss landscape in output space
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('Perturbation in Direction 1')
    ax.set_ylabel('Perturbation in Direction 2')
    ax.set_zlabel('Loss')
    ax.set_title('Loss Landscape as Function of Output Variable')
    fig.colorbar(surf)
    plt.show()

def log_loss_landscape(primal_model, dual_model, loss_fn, dataloader, step):
    # Generate the loss landscape plot
    buf = plot_loss_landscape(primal_model, dual_model, loss_fn, dataloader)

    img = np.array(Image.open(buf))
    wandb_image = wandb.Image(img, caption="Loss Landscape")
    
    # Log the plot to wandb
    wandb.log({
        "loss_landscape": wandb_image,
        "step": step
    })

def primal_loss(trainer, y, eq_cm, ineq_cm, eq_rhs, ineq_rhs, mu, lamb):
        obj = obj_fn_train(trainer, y)

        # ! Alternative penalty: missed demand ** 2
        lagrange_eq = torch.sum(lamb * y[:, trainer.data.md_indices])
        # violation_eq = torch.sum(y[:, self.data.md_indices] ** 2, dim=1)
        violation_eq = torch.sum(y[:, trainer.data.md_indices].abs(), dim=1)
        # rho = trainer.rho
        rho = 126

        penalty = rho/2 * violation_eq

        # ! Primal loss might need to be scaled to work.
        # loss = (obj*1e3 + (lagrange_ineq + lagrange_eq + penalty))
        # loss = (obj + (lagrange_ineq + lagrange_eq + penalty))
        # loss = (obj + (lagrange_eq.clamp(min=0) + penalty))
        loss = (obj*1e3 + lagrange_eq.clamp(min=0) + penalty)

        # ! Test with term regularizing the distance between previous solution.
        # reg = torch.norm(y - self.prev_solution)

        #! Test only optimizing objective.
        # loss = obj

        return loss

def obj_fn_train(trainer, Y):
        # Objective function adjusted for training (different than the actual objective function)
        Y = Y.clone()
        # Y[:, trainer.data.md_indices] = Y[:, trainer.data.md_indices]**2
        Y[:, trainer.data.md_indices] = 0
        # Y[:, self.md_indices] = Y[:, self.md_indices].abs()
        return trainer.data.obj_coeff @ Y.T
        # reg_term_md = torch.norm(Y[:, self.md_indices], p=1, dim=1) #l1 regularization term
        # reg_term_f = torch.norm(Y[:, self.f_lt_indices], p=1, dim=1) #l1 regularization term
        # return self.obj_coeff @ Y.T + 0.1*(reg_term_md + reg_term_f)

def gradient_output(trainer, primal_net, dual_net, loss_fn, dataloader, k=0):
    Xtrain, train_eq_cm, train_ineq_cm, train_eq_rhs, train_ineq_rhs = trainer.X_train, trainer.eq_cm_train, trainer.ineq_cm_train, trainer.eq_rhs_train, trainer.ineq_rhs_train
    outputs = primal_net(Xtrain, train_eq_rhs, train_ineq_rhs)
    outputs.retain_grad()
    if k == 0:
        mu, lamb = torch.zeros_like(train_ineq_rhs), torch.zeros_like(train_eq_rhs)
    else:
        mu, lamb = dual_net(Xtrain, train_eq_cm)
    loss = loss_fn(trainer, outputs, train_eq_cm, train_ineq_cm, train_eq_rhs, train_ineq_rhs, mu, lamb).mean()
    loss.backward()
    print("-"*5, "After Repairs", "-"*50)
    print(f"Output vector: {outputs.tolist()}")
    print("Gradient vector with respect to output:", outputs.grad.tolist())
    split_gradients = trainer.data.split_dec_vars_from_Y(outputs.grad)
    print(f"Gradient w.r.t. Generation: {split_gradients[0].tolist()}")
    print(f"Gradient w.r.t. Line Flows: {split_gradients[1].tolist()}")
    print(f"Gradient w.r.t. Missed Demand: {split_gradients[2].tolist()}")
    

def gradient_before_repairs(trainer, primal_net, dual_net, loss_fn, dataloader, k=0, steps=100):
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    Xtrain = trainer.X_train
    train_eq_cm = trainer.eq_cm_train
    train_ineq_cm = trainer.ineq_cm_train
    train_eq_rhs = trainer.eq_rhs_train
    train_ineq_rhs = trainer.ineq_rhs_train
    primal_optim = torch.optim.Adam(primal_net.parameters(), lr=1e-4)
    # primal_optim = torch.optim.NAdam(primal_net.parameters(), lr=1e-4)

    # Lists to store mean gradients and generation outputs per step
    before_grad_gen = []
    before_grad_line = []
    after_grad_gen = []
    after_grad_line = []
    gen_after_values = []  # Store generation outputs after bound repair

    for step in range(steps):
        primal_optim.zero_grad()
        x_out = primal_net.feed_forward(Xtrain)
        ui_g, p_gt, f_lt = trainer.data.split_dec_vars_from_Y_raw(x_out)
        p_gt.retain_grad(), f_lt.retain_grad()
        
        # [B, bounds, T]
        p_gt_lb, p_gt_ub, f_lt_lb, f_lt_ub, md_nt_lb, md_nt_ub = trainer.data.split_ineq_constraints(train_ineq_rhs)

        p_gt_bound_repaired = primal_net.bound_repair_layer(p_gt, p_gt_lb, p_gt_ub)
        # For line flows, note the lower bound is negative.
        f_lt_bound_repaired = primal_net.bound_repair_layer(f_lt, -f_lt_lb, f_lt_ub)

        p_gt_bound_repaired.retain_grad(), f_lt_bound_repaired.retain_grad()

        UI_g, D_nt = trainer.data.split_eq_constraints(train_eq_rhs)
        md_nt = primal_net.estimate_slack_layer(p_gt_bound_repaired, f_lt_bound_repaired, D_nt)

        y = torch.cat([p_gt_bound_repaired, f_lt_bound_repaired, md_nt], dim=1)\
                 .permute(0, 2, 1).reshape(x_out.shape[0], -1)

        if trainer.data.args["benders_compact"]:
            y = torch.cat([ui_g, y], dim=1)
        if k == 0:
            mu = torch.zeros_like(train_ineq_rhs)
            lamb = torch.zeros_like(train_eq_rhs)
        else:
            mu, lamb = dual_net(Xtrain, train_eq_cm)

        loss = loss_fn(trainer, y, train_eq_cm, train_ineq_cm, train_eq_rhs, train_ineq_rhs, mu, lamb).mean()
        loss.backward()
        
        before_grad_gen.append(p_gt.grad.detach().mean(dim=(0, 2)).cpu().numpy())
        before_grad_line.append(f_lt.grad.detach().mean(dim=(0, 2)).cpu().numpy())
        after_grad_gen.append(p_gt_bound_repaired.grad.detach().mean(dim=(0, 2)).cpu().numpy())
        after_grad_line.append(f_lt_bound_repaired.grad.detach().mean(dim=(0, 2)).cpu().numpy())
        
        # Store the generation output (after bound repair) averaged over batch/time dimensions.
        gen_after = p_gt_bound_repaired.detach().mean(dim=(0, 2))
        gen_after_values.append(gen_after.cpu().numpy())
        if step > 0:
            print(f"New Generation: {p_gt_bound_repaired.flatten().tolist()}")
            print(f"New Lineflows: {f_lt_bound_repaired.flatten().tolist()}")

        if step < steps - 1:
            print("-" * 5, "Step: ", step, "-" * 50)
            print("-" * 5, "Before Repairs", "-" * 50)
            print(f"Output Generation: {p_gt.flatten().tolist()}")
            print(f"Output Lineflows: {f_lt.flatten().tolist()}")
            print(f"Gradient w.r.t. Generation: {p_gt.grad.flatten().tolist()}")
            print(f"Gradient w.r.t. Line Flows: {f_lt.grad.flatten().tolist()}")

            print("-" * 5, "After Bound Repairs", "-" * 50)
            print(f"Gradient w.r.t. Generation: {p_gt_bound_repaired.grad.flatten().tolist()}")
            print(f"Gradient w.r.t. Line Flows: {f_lt_bound_repaired.grad.flatten().tolist()}")
            print(f"Output Generation: {p_gt_bound_repaired.flatten().tolist()}")
            print(f"Output Lineflows: {f_lt_bound_repaired.flatten().tolist()}")
            primal_optim.step()
    
    # Convert lists to numpy arrays for plotting
    before_grad_gen = np.array(before_grad_gen)
    after_grad_gen = np.array(after_grad_gen)
    before_grad_line = np.array(before_grad_line)
    after_grad_line = np.array(after_grad_line)
    gen_after_values = np.array(gen_after_values)
    
    steps_range = list(range(steps))
    num_gen = before_grad_gen.shape[1]
    num_line = before_grad_line.shape[1]
    
    # Create a figure with a gridspec layout: two subplots on the top row and one spanning the full bottom row.
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2])
    
    # Top-left subplot: Gradients per generator
    ax1 = fig.add_subplot(gs[0, 0])
    for i in range(num_gen):
        ax1.plot(steps_range, before_grad_gen[:, i], label=f"Gen {i+1} Before")
        ax1.plot(steps_range, after_grad_gen[:, i], linestyle="--", label=f"Gen {i+1} After")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Mean Gradient (Generation)")
    ax1.set_title("Gradient per Generator Over Time")
    ax1.legend(fontsize="small")
    
    # Top-right subplot: Gradients per line flow
    ax2 = fig.add_subplot(gs[0, 1])
    for i in range(num_line):
        ax2.plot(steps_range, before_grad_line[:, i], label=f"Line {i+1} Before")
        ax2.plot(steps_range, after_grad_line[:, i], linestyle="--", label=f"Line {i+1} After")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Mean Gradient (Line Flows)")
    ax2.set_title("Gradient per Line Flow Over Time")
    ax2.legend(fontsize="small")
    
    # Bottom subplot: Generation outputs (after bound repair) per generator
    ax3 = fig.add_subplot(gs[1, :])
    for i in range(num_gen):
        ax3.plot(steps_range, gen_after_values[:, i], label=f"Gen {i+1}")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Generation Output (After Bound Repair)")
    ax3.set_title("Generation per Generator Over Time")
    ax3.legend(fontsize="small")
    ax3.set_yscale('log')
    
    plt.tight_layout()
    plt.show()

def computation_graph(trainer, primal_net):
    x = trainer.X_train[:1]
    eq_rhs = trainer.eq_rhs_train[:1]
    ineq_rhs = trainer.ineq_rhs_train[:1]
    y = primal_net(x, eq_rhs, ineq_rhs)

    dot = make_dot(y, params=dict(primal_net.named_parameters()))
    dot.format = 'png'
    dot.render('computation_graph')
    
    

def forward(self, x, eq_rhs, ineq_rhs):
    x_out = self.feed_forward(x)

    # [B, G, T], [B, L, T]
    ui_g, p_gt, f_lt = self._data.split_dec_vars_from_Y_raw(x_out)
    
    # [B, bounds, T]
    p_gt_lb, p_gt_ub, f_lt_lb, f_lt_ub, md_nt_lb, md_nt_ub = self._data.split_ineq_constraints(ineq_rhs)

    p_gt_bound_repaired = self.bound_repair_layer(p_gt, p_gt_lb, p_gt_ub)
    # print(p_gt)
    # print(p_gt_bound_repaired)

    # Lineflow lower bound is negative.
    f_lt_bound_repaired = self.bound_repair_layer(f_lt, -f_lt_lb, f_lt_ub)

    UI_g, D_nt = self._data.split_eq_constraints(eq_rhs)
    md_nt = self.estimate_slack_layer(p_gt_bound_repaired, f_lt_bound_repaired, D_nt)

    y = torch.cat([p_gt_bound_repaired, f_lt_bound_repaired, md_nt], dim=1).permute(0, 2, 1).reshape(x_out.shape[0], -1)

    if self._data.args["benders_compact"]:
        y = torch.cat([ui_g, y], dim=1)

    return y
    

if __name__ == "__main__":
    import json

    ## Step 1: parse the input data
    print("Parsing the config file")

    data = parse_config("config.toml")
    experiment = data["experiment"]
    outputs_config = data["outputs_config"]

    with open("config.json", "r") as file:
        args = json.load(file)
    
    print(args)

    # Train the model:
    for i, experiment_instance in enumerate(experiment["experiments"]):
        # Setup output dataframe
        df_res = pd.DataFrame(columns=["setup_time", "presolve_time", "barrier_time", "crossover_time", "restore_time", "objective_value"])

        for j in range(experiment["repeats"]):
            # Run one experiment for j repeats
            run_name = f"refactored_train:{args['train']}_rho:{args['rho']}_rhomax:{args['rho_max']}_alpha:{args['alpha']}_L:{args['alpha']}"
            save_dir = os.path.join('outputs', 'PDL',
                run_name + "-" + str(time.time()).replace('.', '-'))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(os.path.join(save_dir, 'args.dict'), 'wb') as f:
                pickle.dump(args, f)

            target_path = f"outputs/Gurobi/Operational={args['operational']}_T={args['sample_duration']}_Scale={args['scale_problem']}_{args['G']}_{args['L']}"

            # Prep problem data:
            data = prep_data(args=args, inputs=experiment_instance, target_path=target_path, operational=args['operational'])

            model_dir = "outputs/PDL/refactored_train:0.002_rho:0.1_rhomax:5000_alpha:2_L:2-1742895105-291575"

            # Run PDL
            trainer = PrimalDualTrainer(data, args, save_dir, log=False)
            # primal_net, dual_net, stats = trainer.train_PDL()
            primal_net, dual_net = load(args, data, model_dir)

            # wandb.init(project="loss_landscape", entity="test")
            # log_loss_landscape(trainer.primal_net, trainer.dual_net, trainer.primal_loss, trainer.train_loader, step=0)
            # plot_loss_landscape(trainer, primal_net, dual_net, primal_loss, trainer.train_loader, k=args["outer_iterations"])
            # plot_loss_landscape_output(trainer, primal_net, dual_net, primal_loss, trainer.train_loader, k=args["outer_iterations"])
            gradient_before_repairs(trainer, primal_net, dual_net, primal_loss, trainer.train_loader, k=args["outer_iterations"])
            # computation_graph(trainer, primal_net)
