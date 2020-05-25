# ==============================================================================
# analyses
# ==============================================================================

import pandas as pd
from pathlib import Path
# import topsbm_repo.sbmtm # run ; not necessary yet
from graph_tool.all import *
from topsbm import TopSBM
from sklearn.feature_extraction.text import CountVectorizer

# import code.helpers


# data load ====================================================================

# set data path
data_path = Path('/Users/brandonsepulvado/Documents/topsbm_preprocessing/input/data_ethics.xlsx')

# import data
data_ethics = pd.read_excel(data_path)

# get texts
abstracts = data_ethics['AB'].tolist()

# preprocessing ================================================================

# baseline tests
texts_0 = preprocess(abstracts, to_lower=True, rm_stops=False, rm_punct=True, lemmatize=False)
test = [' '.join(item) for item in texts_0]
vec = CountVectorizer(token_pattern=r'\S+')
X = vec.fit_transform(test)
model = TopSBM(random_state=9)
Xt = model.fit_transform(X)
model.plot_graph(n_edges=1000, filename=Path('/Users/brandonsepulvado/Documents/topsbm_preprocessing/output/model_2.png'))


texts_1 = preprocess(abstracts, to_lower=True, rm_stops=True, rm_punct=True, lemmatize=False)

texts_2 = preprocess(abstracts, to_lower=True, rm_stops=True, rm_punct=True, lemmatize=True)

# analyze topical structure ====================================================

model_0 = run_sbmtm(texts_0)
model_1 = run_sbmtm(texts_1)
model_2 = run_sbmtm(texts_2)

gt.vertex_hist(model_2.g, 'total')

model_2.topics(l=1, n=20)
model_2.plot(filename=Path('/Users/brandonsepulvado/Documents/topsbm_preprocessing/output/model_2.png'), nedges=1000)





from graph_tool.all import graph_draw,Graph

#create your graph object
g = Graph()

#add vertex
vertex_1 = g.add_vertex() #here you create a vertex
vertex_2 = g.add_vertex() #here you create a vertex

#add edge
g.add_edge(vertex_1,vertex_2) #add an edge

#draw you graph
graph_draw(
    g,
    output="test.png"
)