# ==============================================================================
# analyses
# ==============================================================================

import pandas as pd
from pathlib import Path
from graph_tool.all import *
from topsbm import TopSBM
from sklearn.feature_extraction.text import CountVectorizer

# data load ====================================================================

# set data path
data_path = Path('/home/brandon/Documents/topsbm_preprocessing/input/data_ethics.xlsx')

# import data
data_ethics = pd.read_excel(data_path)

# get texts
abstracts = data_ethics['AB'].tolist()

# baseline model ===============================================================

# baseline tests

# # preprocess
texts_0 = preprocess(abstracts, to_lower=True, rm_stops=False, rm_punct=True, lemmatize=False)

# prepare and estimate model
vec = CountVectorizer(token_pattern=r'\S+')
X = vec.fit_transform(test)
model = TopSBM(random_state=1234)
Xt = model.fit_transform(X)

# get word probs per topic
topics_model0_l1 = pd.DataFrame(model.groups_[1]['p_w_tw'], index=vec.get_feature_names())
topics_model0_l0 = pd.DataFrame(model.groups_[0]['p_w_tw'], index=vec.get_feature_names())
topics_model0_l2 = pd.DataFrame(model.groups_[2]['p_w_tw'], index=vec.get_feature_names())

# plot
model.state_.draw(layout='bipartite', subsample_edges=1000, hshortcuts=1, hide=0, hvertex_size=5, 
    output = '/home/brandon/Documents/topsbm_preprocessing/output/model_0.pdf')

# get number of words and docs
model.n_features_ # words : 11970
model.n_samples_ # documents : 574
# n_docs = len([v for v in model.graph_.vertices() if model.graph_.vp['kind'][v]==0])

# get top 10 words for topics in level 0
top_ten_model_0_l0 = []
for topic in topics_model0_l0.columns:
    print(topics_model0_l0[topic].nlargest(10))
    top_ten_model_0_l0.append(topics_model0_l0[topic].nlargest(10))

# save
writer=pd.ExcelWriter('/home/brandon/Documents/topsbm_preprocessing/output/model0_l0.xlsx')
for i, A in enumerate(top_ten_model_0_l0):
    A.to_excel(writer,sheet_name=f"topic_{i}")
writer.save()

# get top ten words for topics in level 1
top_ten_model_0_l1 = []
for topic in topics_model0_l1.columns:
    print(topics_model0_l1[topic].nlargest(10))
    top_ten_model_0_l1.append(topics_model0_l1[topic].nlargest(10))

# save
writer=pd.ExcelWriter('/home/brandon/Documents/topsbm_preprocessing/output/model0_l1.xlsx')
for i, A in enumerate(top_ten_model_0_l1):
    A.to_excel(writer,sheet_name=f"topic_{i}")
writer.save()

top_ten_model_0_l2 = []
for topic in topics_model0_l2.columns:
    print(topics_model0_l2[topic].nlargest(10))
    top_ten_model_0_l2.append(topics_model0_l2[topic].nlargest(10))

# save
writer=pd.ExcelWriter('/home/brandon/Documents/topsbm_preprocessing/output/model0_l2.xlsx')
for i, A in enumerate(top_ten_model_0_l2):
    A.to_excel(writer,sheet_name=f"topic_{i}")
writer.save()

# mild preprocessing ===========================================================

# remove stops 
texts_1 = preprocess(abstracts, to_lower=True, rm_stops=True, rm_punct=True, lemmatize=False)

# estimate model
vec = CountVectorizer(token_pattern=r'\S+')
X = vec.fit_transform(test)
model = TopSBM(random_state=1234)
Xt = model.fit_transform(X)

# visually inspect
model.state_.draw(layout='bipartite', subsample_edges=1000, hshortcuts=1, hide=0, hvertex_size=5, 
    output = '/home/brandon/Documents/topsbm_preprocessing/output/model_1.pdf')

# graph description
model.n_features_ # 11716
model.n_samples_ # 574

# get word probs for each level
topics_model1_l1 = pd.DataFrame(model.groups_[1]['p_w_tw'], index=vec.get_feature_names())
topics_model1_l0 = pd.DataFrame(model.groups_[0]['p_w_tw'], index=vec.get_feature_names())
topics_model1_l2 = pd.DataFrame(model.groups_[2]['p_w_tw'], index=vec.get_feature_names())

# get top 10 words for topics in level 0
top_ten_model_1_l0 = []
for topic in topics_model1_l0.columns:
    print(topics_model1_l0[topic].nlargest(10))
    top_ten_model_1_l0.append(topics_model1_l0[topic].nlargest(10))

# save
writer=pd.ExcelWriter('/home/brandon/Documents/topsbm_preprocessing/output/model1_l0.xlsx')
for i, A in enumerate(top_ten_model_1_l0):
    A.to_excel(writer,sheet_name=f"topic_{i}")
writer.save()

# get top ten words for topics in level 1
top_ten_model_1_l1 = []
for topic in topics_model1_l1.columns:
    print(topics_model1_l1[topic].nlargest(10))
    top_ten_model_1_l1.append(topics_model1_l1[topic].nlargest(10))

# save
writer=pd.ExcelWriter('/home/brandon/Documents/topsbm_preprocessing/output/model1_l1.xlsx')
for i, A in enumerate(top_ten_model_1_l1):
    A.to_excel(writer,sheet_name=f"topic_{i}")
writer.save()

# get top ten words in level 2 topics
top_ten_model_1_l2 = []
for topic in topics_model1_l2.columns:
    print(topics_model1_l2[topic].nlargest(10))
    top_ten_model_1_l2.append(topics_model1_l2[topic].nlargest(10))

# save
writer=pd.ExcelWriter('/home/brandon/Documents/topsbm_preprocessing/output/model1_l2.xlsx')
for i, A in enumerate(top_ten_model_1_l2):
    A.to_excel(writer,sheet_name=f"topic_{i}")
writer.save()


# most preprocessing ===========================================================

# remove stops 
texts_2 = preprocess(abstracts, to_lower=True, rm_stops=True, rm_punct=True, lemmatize=True)

# estimate model
vec = CountVectorizer(token_pattern=r'\S+')
X = vec.fit_transform(test)
model = TopSBM(random_state=1234)
Xt = model.fit_transform(X)

# visually inspect
model.state_.draw(layout='bipartite', subsample_edges=1000, hshortcuts=1, hide=0, hvertex_size=5, 
    output = '/home/brandon/Documents/topsbm_preprocessing/output/model_2.pdf')

# graph description
model.n_features_ # 9698
model.n_samples_ # 574

# get word probs for each level
topics_model2_l1 = pd.DataFrame(model.groups_[1]['p_w_tw'], index=vec.get_feature_names())
topics_model2_l0 = pd.DataFrame(model.groups_[0]['p_w_tw'], index=vec.get_feature_names())
topics_model2_l2 = pd.DataFrame(model.groups_[2]['p_w_tw'], index=vec.get_feature_names())

# get top 10 words for topics in level 0
top_ten_model_2_l0 = []
for topic in topics_model2_l0.columns:
    print(topics_model2_l0[topic].nlargest(10))
    top_ten_model_2_l0.append(topics_model2_l0[topic].nlargest(10))

# save
writer=pd.ExcelWriter('/home/brandon/Documents/topsbm_preprocessing/output/model2_l0.xlsx')
for i, A in enumerate(top_ten_model_2_l0):
    A.to_excel(writer,sheet_name=f"topic_{i}")
writer.save()

# get top ten words for topics in level 1
top_ten_model_2_l1 = []
for topic in topics_model2_l1.columns:
    print(topics_model2_l1[topic].nlargest(10))
    top_ten_model_2_l1.append(topics_model2_l1[topic].nlargest(10))

# save
writer=pd.ExcelWriter('/home/brandon/Documents/topsbm_preprocessing/output/model2_l1.xlsx')
for i, A in enumerate(top_ten_model_2_l1):
    A.to_excel(writer,sheet_name=f"topic_{i}")
writer.save()

# get top ten words for topics in level 2
top_ten_model_2_l2 = []
for topic in topics_model2_l2.columns:
    print(topics_model2_l2[topic].nlargest(10))
    top_ten_model_2_l2.append(topics_model2_l2[topic].nlargest(10))

# save
writer=pd.ExcelWriter('/home/brandon/Documents/topsbm_preprocessing/output/model2_l2.xlsx')
for i, A in enumerate(top_ten_model_2_l2):
    A.to_excel(writer,sheet_name=f"topic_{i}")
writer.save()

# # how many docs belong to each topic
# pd.DataFrame(model.groups_[0]['p_tw_d'])
# pd.DataFrame(model.groups_[1]['p_tw_d'])
# model2_l2_topic_counts = pd.DataFrame(model.groups_[2]['p_tw_d']).transpose()
# model2_l2_topic_counts['ident'] = model2_l2_topic_counts.index
# model2_l2_topic_counts = model2_l2_topic_counts.melt(id_vars=['ident'], var_name=['topic'])

# grouped = model2_l2_topic_counts.groupby(by = ['ident'])
# grouped.apply(lambda g: g[g.value == g.value.max()])

# model2_l2_topic_counts.groupby('ident').filter(lambda x : x['value'].max() == x.value)
# model2_l2_topic_counts.loc[model2_l2_topic_counts.groupby(["ident"])["value"].idxmax()]

# idx = model2_l2_topic_counts.groupby(['ident'])['value'].transform(max) == model2_l2_topic_counts['value']
# model2_l2_topic_counts[idx]
# # number of docs per topic
# model2_l2_topic_counts[idx].topic.value_counts().sort_index()

# # look at counts per documnt
# model2_l2_topic_counts[idx].groupby('ident')['topic'].nunique().max()


get_topic_counts(model, 0)
get_topic_counts(model, 1)
get_topic_counts(model, 2)
