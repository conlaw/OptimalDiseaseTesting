# OptimalDiseaseTesting
This project looks at the problem of optimal disease testing on a social network graph. The code is organized as follows

- `optimal_policy.py` - Contains code to compute the optimal policy based on bellman's recursion (note that this is computationally intractable for large problem sizes)
- `heuristics.py` - Outlines all 7 heuristics for testing (suitable for larger problem instances)
- `simulation_helpers.py` - Includes helper functions for simulating a pandemic on an undirected graph
- `university_sim_helpers.py` - Contains code to simulate a realistic graph for a university
