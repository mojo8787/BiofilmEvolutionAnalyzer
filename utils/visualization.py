import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import networkx as nx

def create_heatmap(data_matrix, title="Heatmap"):
    """
    Create a heatmap visualization using Plotly.
    
    Parameters:
    -----------
    data_matrix : pandas.DataFrame
        Data matrix to visualize as a heatmap.
    title : str
        Title for the heatmap.
    
    Returns:
    --------
    plotly.graph_objects.Figure
        The heatmap figure.
    """
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=data_matrix.values,
        x=data_matrix.columns,
        y=data_matrix.index,
        colorscale='RdBu_r',
        zmid=0
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Features",
        yaxis_title="Features",
        height=600,
        width=800
    )
    
    return fig

def generate_pca_plot(data, n_components=2, scale_data=True, group_col=None, full_data=None):
    """
    Generate a PCA plot for dimensionality reduction.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Numeric data for PCA.
    n_components : int
        Number of PCA components to compute.
    scale_data : bool
        Whether to scale the data before PCA.
    group_col : str, optional
        Column name in full_data to use for coloring points.
    full_data : pandas.DataFrame, optional
        Complete dataset including group column.
    
    Returns:
    --------
    plotly.graph_objects.Figure
        The PCA plot.
    """
    # Prepare data for PCA
    X = data.values
    
    # Scale data if requested
    if scale_data:
        X = StandardScaler().fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X)
    
    # Create dataframe with PCA results
    pca_df = pd.DataFrame(
        data=components,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    # Add sample IDs
    pca_df['Sample'] = data.index
    
    # Add group column if provided
    if group_col is not None and full_data is not None:
        # Ensure the group column exists in full_data
        if group_col in full_data.columns:
            # Map group values to samples
            group_map = {}
            for idx, row in full_data.iterrows():
                if idx in data.index:
                    group_map[idx] = row[group_col]
            
            # Add group values to PCA dataframe
            pca_df['Group'] = pca_df['Sample'].map(group_map)
        else:
            print(f"Warning: Group column '{group_col}' not found in data.")
    
    # Create PCA plot
    if n_components >= 3 and 'PC3' in pca_df.columns:
        # Create 3D scatter plot
        if 'Group' in pca_df.columns:
            fig = px.scatter_3d(
                pca_df, x='PC1', y='PC2', z='PC3',
                color='Group', hover_name='Sample',
                title=f'PCA (explained variance: PC1={pca.explained_variance_ratio_[0]:.2f}, '
                      f'PC2={pca.explained_variance_ratio_[1]:.2f}, '
                      f'PC3={pca.explained_variance_ratio_[2]:.2f})'
            )
        else:
            fig = px.scatter_3d(
                pca_df, x='PC1', y='PC2', z='PC3',
                hover_name='Sample',
                title=f'PCA (explained variance: PC1={pca.explained_variance_ratio_[0]:.2f}, '
                      f'PC2={pca.explained_variance_ratio_[1]:.2f}, '
                      f'PC3={pca.explained_variance_ratio_[2]:.2f})'
            )
    else:
        # Create 2D scatter plot
        if 'Group' in pca_df.columns:
            fig = px.scatter(
                pca_df, x='PC1', y='PC2',
                color='Group', hover_name='Sample',
                title=f'PCA (explained variance: PC1={pca.explained_variance_ratio_[0]:.2f}, '
                      f'PC2={pca.explained_variance_ratio_[1]:.2f})'
            )
        else:
            fig = px.scatter(
                pca_df, x='PC1', y='PC2',
                hover_name='Sample',
                title=f'PCA (explained variance: PC1={pca.explained_variance_ratio_[0]:.2f}, '
                      f'PC2={pca.explained_variance_ratio_[1]:.2f})'
            )
    
    # Update layout
    fig.update_layout(
        height=600,
        width=800,
        xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
        yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.2%})'
    )
    
    if n_components >= 3 and 'PC3' in pca_df.columns:
        fig.update_layout(
            scene=dict(
                xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
                yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.2%})',
                zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.2%})'
            )
        )
    
    return fig

def create_correlation_network(correlation_matrix, threshold=0.7):
    """
    Create a network visualization of correlations.
    
    Parameters:
    -----------
    correlation_matrix : pandas.DataFrame
        Correlation matrix.
    threshold : float
        Correlation threshold for including edges.
    
    Returns:
    --------
    plotly.graph_objects.Figure
        The network visualization.
    """
    # Create network graph
    G = nx.Graph()
    
    # Add nodes
    for node in correlation_matrix.columns:
        G.add_node(node)
    
    # Add edges for correlations above threshold
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) >= threshold:
                node1 = correlation_matrix.columns[i]
                node2 = correlation_matrix.columns[j]
                corr = correlation_matrix.iloc[i, j]
                G.add_edge(node1, node2, weight=abs(corr), sign=np.sign(corr))
    
    # Use a layout that spreads nodes nicely
    pos = nx.spring_layout(G, seed=42)
    
    # Prepare node positions and attributes
    node_x = []
    node_y = []
    node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color='rgba(255, 182, 193, 0.8)',
            size=15,
            line_width=2
        )
    )
    
    # Prepare edge positions and attributes
    edge_x = []
    edge_y = []
    edge_colors = []
    edge_width = []
    edge_text = []
    
    for node1, node2, data in G.edges(data=True):
        x0, y0 = pos[node1]
        x1, y1 = pos[node2]
        
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        
        # Set color based on sign (positive=blue, negative=red)
        color = 'blue' if data['sign'] > 0 else 'red'
        edge_colors.extend([color, color, color])
        
        # Set width based on correlation strength
        width = data['weight'] * 3
        edge_width.extend([width, width, width])
        
        # Hover text
        text = f"{node1} - {node2}: {data['weight']:.3f}"
        edge_text.extend([text, text, text])
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(
            color=edge_colors,
            width=edge_width
        ),
        hoverinfo='text',
        text=edge_text
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title=f'Correlation Network (threshold = {threshold})',
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20, l=5, r=5, t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       height=600,
                       width=800
                   ))
    
    return fig

def create_volcano_plot(results_df, x_col='log2FoldChange', y_col='padj', title="Volcano Plot"):
    """
    Create a volcano plot for differential expression analysis.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        Results dataframe with log2FoldChange and p-values.
    x_col : str
        Column name for x-axis (log2 fold change).
    y_col : str
        Column name for y-axis (p-value or adjusted p-value).
    title : str
        Plot title.
    
    Returns:
    --------
    plotly.graph_objects.Figure
        The volcano plot.
    """
    # Calculate -log10 of p-values
    log_pval = -np.log10(results_df[y_col])
    
    # Create a new dataframe for plotting
    plot_df = pd.DataFrame({
        'log2FoldChange': results_df[x_col],
        '-log10(pvalue)': log_pval,
        'Gene': results_df.index if results_df.index.name is None else results_df.index.values
    })
    
    # Define significance threshold (e.g., padj < 0.05, |log2FC| > 1)
    is_significant = (results_df[y_col] < 0.05) & (abs(results_df[x_col]) > 1)
    plot_df['Significant'] = is_significant
    
    # Define up/down regulation
    plot_df['Regulation'] = 'Not Significant'
    plot_df.loc[(plot_df['Significant']) & (plot_df['log2FoldChange'] > 0), 'Regulation'] = 'Up-regulated'
    plot_df.loc[(plot_df['Significant']) & (plot_df['log2FoldChange'] < 0), 'Regulation'] = 'Down-regulated'
    
    # Create volcano plot
    fig = px.scatter(
        plot_df, 
        x='log2FoldChange', 
        y='-log10(pvalue)',
        color='Regulation',
        hover_name='Gene',
        color_discrete_map={
            'Up-regulated': 'red',
            'Down-regulated': 'blue',
            'Not Significant': 'gray'
        }
    )
    
    # Add horizontal line for p-value threshold
    fig.add_shape(
        type='line',
        y0=-np.log10(0.05), y1=-np.log10(0.05),
        x0=min(plot_df['log2FoldChange']), x1=max(plot_df['log2FoldChange']),
        line=dict(color='black', dash='dash')
    )
    
    # Add vertical lines for fold change thresholds
    fig.add_shape(
        type='line',
        x0=1, x1=1,
        y0=0, y1=max(plot_df['-log10(pvalue)']),
        line=dict(color='black', dash='dash')
    )
    fig.add_shape(
        type='line',
        x0=-1, x1=-1,
        y0=0, y1=max(plot_df['-log10(pvalue)']),
        line=dict(color='black', dash='dash')
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Log2 Fold Change",
        yaxis_title="-Log10(P-value)",
        height=600,
        width=800
    )
    
    return fig

def create_pathway_enrichment_plot(enrichment_results, title="Pathway Enrichment Analysis"):
    """
    Create a plot for pathway enrichment analysis results.
    
    Parameters:
    -----------
    enrichment_results : pandas.DataFrame
        Enrichment analysis results with pathways and p-values.
    title : str
        Plot title.
    
    Returns:
    --------
    plotly.graph_objects.Figure
        The enrichment plot.
    """
    # Sort by p-value
    sorted_results = enrichment_results.sort_values('pvalue')
    
    # Take top 20 pathways
    top_results = sorted_results.head(20)
    
    # Create bar plot
    fig = px.bar(
        top_results,
        y='pathway',
        x='-log10(pvalue)',
        orientation='h',
        color='-log10(pvalue)',
        color_continuous_scale='Viridis'
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="-Log10(P-value)",
        yaxis_title="Pathway",
        height=600,
        width=800,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def create_regulatory_network_plot(regulators, targets, interactions, title="Regulatory Network"):
    """
    Create a visualization of a gene regulatory network.
    
    Parameters:
    -----------
    regulators : list
        List of regulator genes.
    targets : list
        List of target genes.
    interactions : list of tuples
        List of (regulator, target, effect) tuples, where effect is 1 for activation, -1 for repression.
    title : str
        Plot title.
    
    Returns:
    --------
    plotly.graph_objects.Figure
        The network visualization.
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for reg in regulators:
        G.add_node(reg, type='regulator')
    
    for target in targets:
        G.add_node(target, type='target')
    
    # Add edges
    for reg, target, effect in interactions:
        G.add_edge(reg, target, effect=effect)
    
    # Use a layout that spreads nodes nicely
    pos = nx.spring_layout(G, seed=42)
    
    # Prepare node positions and attributes
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_sizes = []
    
    for node, attr in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        
        # Color by node type
        if attr['type'] == 'regulator':
            node_colors.append('red')
            node_sizes.append(15)
        else:
            node_colors.append('blue')
            node_sizes.append(10)
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line_width=2
        )
    )
    
    # Prepare edge positions and attributes
    edge_x = []
    edge_y = []
    edge_colors = []
    edge_width = []
    
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        
        # Set color based on effect (activation=green, repression=red)
        color = 'green' if data['effect'] > 0 else 'red'
        edge_colors.extend([color, color, color])
        
        # Set width
        width = 2
        edge_width.extend([width, width, width])
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(
            color=edge_colors,
            width=edge_width
        ),
        hoverinfo='none'
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title=title,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20, l=5, r=5, t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       height=600,
                       width=800
                   ))
    
    return fig

def create_enrichment_heatmap(enrichment_matrix, title="Pathway Enrichment Heatmap"):
    """
    Create a heatmap of enrichment results across multiple conditions.
    
    Parameters:
    -----------
    enrichment_matrix : pandas.DataFrame
        Matrix with pathways as rows and conditions as columns, values are -log10(p-value).
    title : str
        Heatmap title.
    
    Returns:
    --------
    plotly.graph_objects.Figure
        The heatmap figure.
    """
    # Create heatmap
    fig = px.imshow(
        enrichment_matrix,
        labels=dict(x="Condition", y="Pathway", color="-log10(p-value)"),
        x=enrichment_matrix.columns,
        y=enrichment_matrix.index,
        color_continuous_scale="Viridis"
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=800,
        width=1000
    )
    
    return fig
