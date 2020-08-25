#basic plot funcs
import chart_studio.plotly as py
from plotly.graph_objs import *
import networkx as nx
import numpy as np

from utils import *

def plot_nodes(nodes, pos, labels, colors):
    
    Xv=[pos[k][0] for k in nodes]
    Yv=[pos[k][1] for k in nodes]
    
    trace_nodes=Scatter(x=Xv,
               y=Yv,
               mode='markers',
               name='net',
               marker=dict(symbol='circle-dot',
                             size=50,
                             color=colors, #list of node colors
                             line=dict(color='rgb(50,50,50)', width=0.5)
                             ),
               text=labels,
               hoverinfo='text'
               )
    return [trace_nodes]


def plot_edges(edges, weights, pos, labels, colors, alpha=1.0):
    
    trace_edge_list = []
    
    middle_node_trace = Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=Marker(
            opacity=0,
            color='grey'
        )
    )
    
    for edge, weight, label, color in zip(edges, weights, labels, colors):
        trace_edge=Scatter(
            x=[],
            y=[],
            mode='lines',
            opacity=alpha,
            line=Line(color=color, width=weight),
            hoverinfo='none'
        )
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        trace_edge['x'] += (x0, x1, None)
        trace_edge['y'] += (y0, y1, None)
        trace_edge_list.append(trace_edge)

        middle_node_trace['x'] += tuple([(x0+x1)/2])
        middle_node_trace['y'] += tuple([(y0+y1)/2])
        middle_node_trace['text'] += tuple([label])
    
    return [*trace_edge_list, middle_node_trace]



def node_labels(nodes, graph):
    res = []
    colors = []
    for node in nodes:
        _meta = graph.nodes()[node]
        _df = _meta['idf']
        _pop = np.log10(_meta['pop'] + 1e-6)
        _rt = _meta['rt']
        _fav = _meta['fav']
        _color = str(_meta['color'])
        _freq = _meta['freq']
        
        _label = f"'name':{node}<br>'df': {_df}<br>'pop': {_pop}<br>'rt': {_rt}<br>'fav': {_fav}<br>'color': {_color}<br>'freq': {_freq}"
        res.append(_label)
        
        colors.append(take_node_color(_meta))
    return res, colors

def edge_labels(edges, graph):
    res = []
    colors = []
    wts = []
    for edge in edges:
        _meta = graph.edges()[edge]
        _wt = _meta['weight']
        wts.append(float(_wt) * 2)
        _sus = _meta['sus']
        _color = str(_meta['color'])
        
        _label = f"'weight': {_wt}<br>'d_suspicious': {_sus}<br>'colors': {_color}"
        res.append(_label)
        
        colors.append(take_node_color(_meta))
    return res, colors, wts

def plot_rumor(name, week, graph, act_mapping, centres, width=1000, height=1000):
    
    
    

    axis=dict(showline=False, # hide axis line, grid, ticklabels and  title
              zeroline=False,
              showgrid=False,
              showticklabels=False,
              title=''
              )
    
    lo=Layout(title= f"The rumor graph of {name} at week {week}",
    font= dict(size=12),
    showlegend=False,
    autosize=False,
    width=width,
    height=height,
    xaxis=layout.XAxis(axis),
    yaxis=layout.YAxis(axis),
    margin=layout.Margin(
        l=40,
        r=40,
        b=85,
        t=100,
    ),
              
    hovermode='closest',
    annotations=[
           dict(
           showarrow=False,
            text='The rumor graph',
            xref='paper',
            yref='paper',
            x=0,
            y=-0.1,
            xanchor='left',
            yanchor='bottom',
            font=dict(
            size=14
            )
            )
        ]
    )
    
    
    #plot nodes
    _pos = nx.fruchterman_reingold_layout(graph)
    _labels, _colors = node_labels(graph.nodes(), graph)
    _nds = plot_nodes(list(graph.nodes()), pos=_pos, labels=_labels, colors=_colors)
    
    
    
    #plot edges with gray color
    _edge_labels, _edge_colors, _edge_wts = edge_labels(graph.edges(), graph)
    
    _eds = plot_edges(list(graph.edges()), weights=_edge_wts, pos=_pos, labels=_edge_labels, colors=_edge_colors, alpha=0.2)
    
    _centre_lst = []
    for centre in centres:
        _centre_lst += any_in(act_mapping[centre], graph)
    _centre_lst = set(_centre_lst)


    _nd_lst = any_in(act_mapping[name], graph)
    
    
    data = []
    
    for _ct in _centre_lst:
        for _nd in _nd_lst:
            _short = get_path_edge(graph, _ct, _nd, weighted=False)
            _unsus = get_path_edge(graph, _ct, _nd, effort='sus', ret='sus')
            _sus = get_path_edge(graph, _ct, _nd, effort='rev_sus', ret='sus')
            _unpop = get_path_edge(graph, _ct, _nd, effort='weight', ret='weight')
            _pop = get_path_edge(graph, _ct, _nd, effort='reverse_w', ret='weight')
            
            
            _plt_st = plot_edges(_short, weights=[2]*len(_short), pos=_pos, labels=['']*len(_short), colors=['black']*len(_short), alpha=1)
            _plt_sus = plot_edges(_sus, weights=[2]*len(_sus), pos=_pos, labels=['']*len(_sus), colors=['purple']*len(_sus), alpha=1)
            _plt_unsus = plot_edges(_unsus, weights=[2]*len(_unsus), pos=_pos, labels=['']*len(_unsus), colors=['green']*len(_unsus), alpha=1)
            
            
            data += _plt_st
            data += _plt_sus
            data += _plt_unsus
#             data += _plt_pop
    
            
#     eds = plot_edges(list(graph.edges()), wts, pos, labels, colors=['red'])
    
    data += _eds
    data += _nds
    
    fig=Figure(data=data, layout=lo)
    
    return fig


