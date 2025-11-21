import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional

def plot_scatter(
    df: pd.DataFrame, 
    x_col: str, 
    y_col: str, 
    hue: Optional[str] = None, 
    title: str = "", 
    out_path: str = "plot.png"
):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, palette="viridis", s=60, alpha=0.7)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_clustering_metrics(history: list, out_path: str):
    df = pd.DataFrame(history)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('k')
    ax1.set_ylabel('Silhouette Score', color=color)
    ax1.plot(df['k'], df['silhouette'], color=color, marker='o')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Davies-Bouldin', color=color)
    ax2.plot(df['k'], df['davies_bouldin'], color=color, marker='x', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("Clustering Metrics vs K")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
