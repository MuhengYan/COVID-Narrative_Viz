import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

from tqdm import tqdm

import pickle


from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()


from LIWC.LIWC_package import liwc
cat, dic = liwc.read_liwc('LIWC/LIWC_package/liwc_data/LIWC2007_English131104.dic')


def sel_lemma(token):
    exclude = ['us', 'u.s.']
    if token in exclude:
        return token
    else:
        return lemmatizer.lemmatize(token)


from nltk.corpus import stopwords
STOP = stopwords.words('english')
STOP += [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
STOP += ['coronavirus']
STOP += [lemmatizer.lemmatize(_tk.lower()) for _tk in STOP]
STOP = set(STOP)

import string  
PUNC = string.punctuation



from nltk.tokenize import NLTKWordTokenizer
tknz = NLTKWordTokenizer()


from scipy import spatial
from sklearn import metrics

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import networkx as nx
import community as community_louvain

from scipy.stats import ttest_ind_from_stats
from scipy import stats

import random
random.seed(42)

import mplcursors


def remove_nums(string):
    tokens = string.split()
    res = []
    for t in tokens:
        if '{' in t and '}' in t:
            
            try:
                float(t.replace('{', '').replace('}', '').replace(',', ''))
                t = '{<<NUM>>}'
            except Exception:
                pass
            
        else:
            try:
                float(t.replace(',', ''))
                t = '<<NUM>>'
            except Exception:
                pass
        res.append(t)
    return ' '.join(res)
    

def read_df_rel(based_dir, file_input_name):
    file_input = based_dir + file_input_name    
    ff = open(file_input)
    delim=","
    df = pd.read_csv(file_input,delimiter=delim,header=0)        
    return df


def find_ngrams(df, n=2):
    ngrams = {}
    _sents = set([])
    
    
    for row in df.iterrows():
        _sent = row[1]['sentence']
        _sents.add(_sent)
    
        
    for _sent in tqdm(_sents):
        _tokens = tknz.tokenize(_sent)
        _tokens = [remove_nums(lemmatizer.lemmatize(_tk.lower())) for _tk in _tokens]
        for i in range(len(_tokens) - n+1):
            _tks = _tokens[i : i+n]
            flag = False
            for _tk in _tks:
                if _tk in STOP:
                    flag = True
                if _tk in PUNC:
                    flag = True
            if flag:
                continue
            _gram = '_'.join(_tks)
            if _gram not in ngrams:
                ngrams[_gram] = 0
            ngrams[_gram] += 1
            
    return ngrams

def find_grams_scores(uni, ngrams, delim='_', threshold=0.5):
    res = {}
    for grams in ngrams:
        gram_score = {}
        for gram in tqdm(grams):
            _tokens = gram.split(delim)
            _freq = grams[gram]
            _union = np.sum([uni[_tk] for _tk in _tokens])
            _score = _freq/_union
            
            flag = False
            if _score >= threshold:
                flag = True
                for g in res:
                    if gram in g:
                        flag = False
            if flag:
                res[gram] = _freq
    return res


    
    
def find_arc_word(arg, merge_grams=None, n=3):
    
    if not ' ' in arg:
        return remove_nums(lemmatizer.lemmatize(arg.lower().replace('{', '').replace('}', '')))
    
    if '{' not in arg and '}' not in arg:
        return remove_nums(lemmatizer.lemmatize(arg.lower().replace('{', '').replace('}', '')))
    
    _potential = ''
    _switch = False
    for _char in arg:
        if _char == '{':
            _switch = True
        if _char == '}':
            _switch = False
        if _switch:
            _potential += _char
            
    if ' ' in _potential:
        arg = arg.replace(_potential, _potential.replace(' ', '_'))
    
    _tokens = arg.lower().split()
    _arc_idx = -1
    for i, _tk in enumerate(_tokens):
        if '{' in _tk and '}' in _tk:
            _arc_idx = i
            _tokens[i] = _tk.replace('{', '').replace('}', '')
    
    
    if _arc_idx == -1:
        return None
    
    
    _tokens = [remove_nums(lemmatizer.lemmatize(_tk.lower())) for _tk in _tokens]
    
    
    if merge_grams is None:
        return _tokens[_arc_idx]
    
    else:
        candidate = []
        
        for _n in range(1, n):
            if not _arc_idx - _n < 0:
                _gram = '_'.join(_tokens[_arc_idx - _n: _arc_idx + 1])
                if _gram in merge_grams:
                    candidate.append(_gram)
            
            if not _arc_idx + _n > len(_tokens):
                _gram = '_'.join(_tokens[_arc_idx : _arc_idx + _n + 1])
                if _gram in merge_grams:
                    candidate.append(_gram)
        
        candidate = set(candidate)
        if len(candidate) == 0:
            return _tokens[_arc_idx]
        elif len(candidate) > 1:
            _sorted = sorted(candidate, key=lambda x:len(x))
            for i in range(len(_sorted)):
                _grams = _sorted[i]
                for j in range(i + 1, len(_sorted)):
                    if _grams in _sorted[j]:
                        candidate.remove(_grams)
            if len(candidate) > 1:
                candidate = sorted(candidate, key=lambda x:merge_grams[x], reverse=True)
                return list(candidate)[0]
            else:
                return list(candidate)[0]
        else:
            return list(candidate)[0]
        
        


            
    


def extract_args(df, merge_grams):
    reverse = ['(nsubjpass, verb (no obj),prep', '(nsubjpass, verb (no obj),prepc']
    not_OK = ['(nsubj, verb, (O)prep)', '(xsubj, verb, noun-cop)', '(srl-A0, srl-v, srl-A2)', '(srl-A0, srl-v, srl-A1)','(nsubj, verb (no obj),prepc',  '(nsubj, verb)', 
         '(xsubj, verb (no obj),prepc', '(nsubjpass, verb, dobj)', '(xsubj, verb)', '(nsubjpass, verb)']

    args = {}
    rels = {}
    triplets = {}
    posts_num = {}
    
    _seen = {}
    
    
    
    for row in tqdm(df.iterrows(), total=df.shape[0]):
        
        _post = row[1]['post_num']
        _sent = row[1]['sentence_num']
        
        _sent_text = row[1]['sentence']
        _text_cats = liwc.get_text2cat(text=_sent_text, dic=dic, cat=cat)
        
        
        _identifier = '-'.join([str(_post), str(_sent)])
        if _identifier not in _seen:
            _seen[_identifier] = set([])
            
        
        
        if row[1]['pattern'] in not_OK:
            continue
        elif row[1]['pattern'] in reverse:
            _arg1 = row[1]['arg2']
            _arg2 = row[1]['arg1']        
        else:
            _arg1 = row[1]['arg1']
            _arg2 = row[1]['arg2']
        
        #appear in one sent, count only once
        _arc1 = find_arc_word(_arg1, merge_grams=merge_grams)
        _arc2 = find_arc_word(_arg2, merge_grams=merge_grams)
        
        _verb = find_arc_word(row[1]['rel'], merge_grams=merge_grams)
        
        _trip = '->'.join([_arc1, _verb, _arc2])
        
        
        
        if _trip in _seen[_identifier]:
            continue
        else:
            _seen[_identifier].add(_trip)
            
            if _arc1 not in args:
                args[_arc1] = []
            if _arc2 not in args:
                args[_arc2] = []
            if _trip not in triplets:
                triplets[_trip] = set([])
            if _verb not in rels:
                rels[_verb] = set([])
            
            args[_arc1].append({'arg': _arg1, 'post': _post, '_sent': _sent, 'LIWC': _text_cats})
            args[_arc2].append({'arg': _arg2, 'post': _post, '_sent': _sent, 'LIWC': _text_cats})
            triplets[_trip].add(_post)
            rels[_verb].add(_post)
        
    return args, triplets, rels

def max_id(dic):
    if len(dic) == 0:
        return 0
    else:
        return max([idx for idx in dic.keys()]) + 1
    

def get_super_node_label(super_node, word_tfidf, k=3):
    
    token_idf = {}
    for token in super_node:
        
        if '_' in token:
            _s = []
            for _t in token.split('_'):
                if _t in word_tfidf:
                    _s.append(word_tfidf[_t])
                    
            if len(_s) == 0:
                _score = 0
            else:
                _score = np.mean(_s)
            
        else:        
            if token in word_tfidf:
                _score = word_tfidf[token]
            else:
                _score = 0
                
        token_idf[token] = _score
        
        
            
    sorted_idf = sorted(token_idf.items(), key=lambda x: x[1], reverse=True)[:k]
    return 'SUPER:'+'|'.join([tk[0] for tk in sorted_idf])


#build graphs

def rec_remove(G, fix_iter=20):
    _size = len(G)
    for it in range(fix_iter):
        
        for node in list(G.nodes):
            if G.degree(node) <= 1:
                G.remove_node(node)
        new_size = len(G)
        if new_size == _size:
            break
        else:
            _size = new_size
    
    return G


def color_mat(posts, raw):
    colors = ['green', 'yellow', 'orange', 'red']
    res = np.zeros(4)
    for _p in posts:
        res[colors.index(raw[_p]['coding'])] += 1.0
    return res


def take_node_color(node_colors):
    colors = ['green', 'yellow', 'orange', 'red']
    
    
    for idx, p in enumerate(node_colors):
        if p >= 0.5:
            return colors[idx]

    return 'black'


def build_graph_perc_filter(nodes, edges, filter_it=-1, directed=False, verbose=False):

    
    discard_rare = set([])
#     for node in nodes:
#         if 'SUPER' in node:
#             continue
#         if nodes[node]['freq'] < 2:
#             discard_rare.add(node)
        
            
            
    # build graph
    if directed:
        G = nx.DiGraph()
        
        for node in nodes:
            if node not in discard_rare:
                G.add_node(node)
                for _feat in nodes[node]:
                    G.nodes()[node][_feat] = nodes[node][_feat]
                    
                
        print(f'before removal of low degree nodes: #nodes={len(G)}')
        
        for edge in edges:
            _from = edge.split('->')[0]
            _to = edge.split('->')[1]
            _weight = np.sum([item['idf'] for item in edges[edge]])
            _name = edge
            
            
#             _scores = []
#             for item in edges[edge]:
#                 _scores += item['suspicious']
            
            _colors = np.zeros(4)
            for item in edges[edge]:
                _colors += item['color']
                
            _colors = _colors/np.sum(_colors)
            
            _verbs = ', '.join(list(set([item['verb'] for item in edges[edge]])))
                
            if _from in G and _to in G:
#                 G.add_edge(_from, _to, weight=_weight, sus=np.mean(_scores), name=_name, color=_colors, reverse_w=1/_weight, rev_sus=1-np.mean(_scores), verbs=_verbs)
                G.add_edge(_from, _to, weight=_weight, name=_name, color=_colors, reverse_w=1/_weight, verbs=_verbs)
                
                
    else:
        G = nx.Graph()
        for node in nodes:
            if node not in discard_rare:
                G.add_node(node)
                for _feat in nodes[node]:
                    G.nodes()[node][_feat] = nodes[node][_feat]
        print(f'before removal of low degree nodes: #nodes={len(G)}')

        
        for edge in edges:
            _from = edge.split('<->')[0]
            _to = edge.split('<->')[1]
            _weight = np.sum([item['idf'] for item in edges[edge]])
            _name = edge
            
#             _scores = []
#             for item in edges[edge]:
#                 _scores += item['suspicious']
                
            _colors = np.zeros(4)
            for item in edges[edge]:
                _colors += item['color']
                
            _colors = _colors/np.sum(_colors)
            
            _verbs = ', '.join(list(set([item['verb'] for item in edges[edge]])))
            if _from in G and _to in G:
#                 G.add_edge(_from, _to, weight=_weight, sus=np.mean(_scores), name=_name, color=_colors, reverse_w=1/_weight, rev_sus=1-np.mean(_scores), verbs=_verbs)
                G.add_edge(_from, _to, weight=_weight, name=_name, color=_colors, reverse_w=1/_weight, verbs=_verbs)
                
        
    if filter_it > 0:
        G = rec_remove(G, fix_iter=filter_it)

    print(f'after removal of low degree nodes: #nodes={len(G)}')

    
    return G

def any_in(nodes, graph):
    res = []
    for node in nodes:
        if node in graph:
            res.append(node)
    return res


def show_shortest_path(graph, word1, word2, weighted=True, effort='sus', show='sus'):
    
    try:
        if weighted:
            nodes = nx.algorithms.shortest_path(graph, word1, word2, weight=effort)
        else:
            nodes = nx.algorithms.shortest_path(graph, word1, word2)
    except Exception:
        return []
        
    _efforts = []
    for i in range(len(nodes) - 1):
        _pair = (nodes[i], nodes[i + 1])
        _edge = graph.edges[_pair]
        _effort = _edge[show]
        _efforts.append(_effort)
        

    return nodes


def get_path_effort(graph, word1, word2, weighted=True, effort='sus', ret='sus'):
    
    if weighted:
        nodes = nx.algorithms.shortest_path(graph, word1, word2, weight=effort)
    else:
        nodes = nx.algorithms.shortest_path(graph, word1, word2)
        
    _efforts = []
    for i in range(len(nodes) - 1):
        _pair = (nodes[i], nodes[i + 1])
        _edge = graph.edges[_pair]
        _effort = _edge[ret]
        _efforts.append(_effort)
    
    if not weighted:
        return len(nodes) - 1
    else:
        return np.mean(_efforts)
    
def get_path_edge(graph, word1, word2, weighted=True, effort='sus', ret='sus'):
    
    if weighted:
        nodes = nx.algorithms.shortest_path(graph, word1, word2, weight=effort)
    else:
        nodes = nx.algorithms.shortest_path(graph, word1, word2)
        
    if len(nodes) > 1:
        _edges = []
        for i in range(len(nodes) - 1):
            _edges.append(tuple([nodes[i], nodes[i + 1]]))
        return _edges
    else:
        return None

def get_average_effort(graph, list1, list2, weighted=True, effort='sus', ret='sus'):
    
    res = []
    for nd1 in list1:
        for nd2 in list2:
            _eff = get_path_effort(graph, nd1, nd2, weighted=weighted, effort=effort, ret=ret)
            
            res.append(_eff)

    return np.mean(res)
