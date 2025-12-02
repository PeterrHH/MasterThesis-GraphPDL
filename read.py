import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------
# 1. Toy graph + data
# ----------------------------

# Simple 3-node undirected graph: 0--1--2
# Adjacency matrix (including self-loops)
A = torch.tensor([
    [1., 1., 0.],
    [1., 1., 1.],
    [0., 1., 1.]
])  # shape (3, 3)

# Normalize adjacency (very simple degree normalization)
deg = A.sum(dim=1, keepdim=True)  # degree of each node
A_norm = A / deg                  # row-normalized adjacency

# Node features: [demand, cost]
# Say:
# node 0: demand 1.0, cost 1.0
# node 1: demand 2.0, cost 2.0
# node 2: demand 1.5, cost 3.0
X = torch.tensor([
    [1.0, 1.0],
    [2.0, 2.0],
    [5.5, 3.0]
])  # shape (3, 2)

demand = X[:, 0]  # first feature is demand
cost   = X[:, 1]  # second feature is cost

# ----------------------------
# 2. Simple GNN model
# ----------------------------

class SimpleGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)  # output: generation per node

    def forward(self, x, A_norm):
        """
        x: node features (N, in_dim)
        A_norm: normalized adjacency (N, N)
        """
        # Message passing: aggregate neighbors
        h = A_norm @ x             # (N, in_dim)
        h = torch.relu(self.lin1(h))
        gen = self.lin2(h).squeeze(-1)  # (N,)
        # Ensure non-negative generation
        gen = torch.relu(gen)
        return gen  # predicted generation at each node


model = SimpleGNN(in_dim=2, hidden_dim=8)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ----------------------------
# 3. Self-supervised loss
# ----------------------------
# No ground truth generation.
# We only know:
#  - total generation should match total demand
#  - we want to minimize cost * generation
# So:
#   loss = total_cost + penalty_balance

def compute_loss(gen, demand, cost, balance_weight=10.0):
    """
    gen: predicted generation per node (N,)
    demand, cost: known vectors (N,)
    """
    total_gen = gen.sum()
    total_dem = demand.sum()

    # Economic cost
    total_cost = (cost * gen).sum()

    # Power balance penalty: (sum gen - sum demand)^2
    balance_penalty = (total_gen - total_dem) ** 2

    loss = total_cost + balance_weight * balance_penalty
    return loss, total_cost.item(), balance_penalty.item()

# ----------------------------
# 4. Training loop (self-supervised)
# ----------------------------

for epoch in range(500):
    optimizer.zero_grad()
    gen = model(X, A_norm)                          # predicted generation per node
    loss, total_cost, balance_penalty = compute_loss(gen, demand, cost)

    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch:03d} | "
              f"Loss: {loss.item():.4f} | "
              f"Cost: {total_cost:.4f} | "
              f"Balance penalty: {balance_penalty:.4f} | "
              f"Total gen: {gen.sum().item():.3f}, Total dem: {demand.sum().item():.3f}")

# ----------------------------
# 5. Inspect final solution
# ----------------------------
gen_final = model(X, A_norm).detach()
print("\nFinal predicted generation per node:")
for i, g in enumerate(gen_final):
    print(f"  Node {i}: gen = {g.item():.3f}, demand = {demand[i].item():.3f}, cost = {cost[i].item():.3f}")

print(f"\nTotal generation: {gen_final.sum().item():.3f}")
print(f"Total demand:     {demand.sum().item():.3f}")
