import matplotlib.pyplot as plt
import networkx as nx

# Parse the .dot file and create the graph
graph = nx.drawing.nx_pydot.read_dot("dag.dot")

# Draw the graph
plt.figure(figsize=(12, 8))
nx.draw(
    graph,
    with_labels=True,
    node_size=1500,
    node_color="lightblue",
    font_size=10,
    font_weight="bold",
)
plt.title("Snakemake DAG", fontsize=16)

# Save the plot
plt.savefig("plots/dag.png", dpi=300, format="png")

