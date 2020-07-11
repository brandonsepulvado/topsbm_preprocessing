# ==============================================================================
# analyses
# ==============================================================================

import pandas as pd
from pathlib import Path
from graph_tool.all import *
from topsbm import TopSBM
from sklearn.feature_extraction.text import CountVectorizer
import random

# data load ====================================================================

# set data path/
path_tweets = Path('/mnt/h/topsbm_preprocessing/data/timeline_thought_leaders.csv')

# import data
data_tweets = pd.read_csv(path_tweets)

# get texts
texts = data_tweets['bodypost'].dropna()

# set seed
seed_val = 1234

# initialize seed
random.seed(seed_val)

# # number of list elements to retain
n_docs = 10000

# reworking because want to get set of indices to be able to reproduce
indices_10k = random.sample(range(0, len(texts), 1), n_docs)

tweets = texts[texts.index.isin(indices_10k)].reset_index()

tweets_list = tweets['bodypost'].to_list()

texts_0 = preprocess(tweets_list, to_lower=True, rm_stops=False, rm_punct=True, lemmatize=False)

model_0, vec_0 = est_model(texts_0)

# get word probs per topic
topics_model0_l1 = pd.DataFrame(model_0.groups_[1]['p_w_tw'], index=vec.get_feature_names())
topics_model0_l0 = pd.DataFrame(model_0.groups_[0]['p_w_tw'], index=vec.get_feature_names())
topics_model0_l2 = pd.DataFrame(model_0.groups_[2]['p_w_tw'], index=vec.get_feature_names())

# plot
model_0.state_.draw(layout='bipartite', subsample_edges=1000, hshortcuts=1, hide=0, hvertex_size=5, 
    output = '/mnt/h/topsbm_preprocessing/tweets_model_0.pdf')
    
# get number of words and docs
words_m0 = model_0.n_features_ # words : 43742
docs_m0 = model_0.n_samples_ # documents : 10000

# # get top 10 words for topics in level 0
# top_ten_model_0_l0 = []
# for topic in topics_model0_l0.columns:
#     print(topics_model0_l0[topic].nlargest(10))
#     top_ten_model_0_l0.append(topics_model0_l0[topic].nlargest(10))

# # save
# writer=pd.ExcelWriter('/mnt/h/topsbm_preprocessing/tweets_model0_l0.xlsx')
# for i, A in enumerate(top_ten_model_0_l0):
#     A.to_excel(writer,sheet_name=f"topic_{i}")
# writer.save()

# # get top ten words for topics in level 1
# top_ten_model_0_l1 = []
# for topic in topics_model0_l1.columns:
#     print(topics_model0_l1[topic].nlargest(10))
#     top_ten_model_0_l1.append(topics_model0_l1[topic].nlargest(10))

# # save
# writer=pd.ExcelWriter('/home/brandon/Documents/topsbm_preprocessing/output/model0_l1.xlsx')
# for i, A in enumerate(top_ten_model_0_l1):
#     A.to_excel(writer,sheet_name=f"topic_{i}")
# writer.save()

# top_ten_model_0_l2 = []
# for topic in topics_model0_l2.columns:
#     print(topics_model0_l2[topic].nlargest(10))
#     top_ten_model_0_l2.append(topics_model0_l2[topic].nlargest(10))

# # save
# writer=pd.ExcelWriter('/home/brandon/Documents/topsbm_preprocessing/output/model0_l2.xlsx')
# for i, A in enumerate(top_ten_model_0_l2):
#     A.to_excel(writer,sheet_name=f"topic_{i}")
# writer.save()

# mid-preproc ==================================================================

texts_1 = preprocess(tweets_list, to_lower=True, rm_stops=True, rm_punct=True, lemmatize=False)

model_1, vec_1 = est_model(texts_1)

# number of topic groups
len(model_1.groups_) # 1

# # get word probs per topic
# topics_model0_l1 = pd.DataFrame(model_0.groups_[1]['p_w_tw'], index=vec.get_feature_names())
# topics_model0_l0 = pd.DataFrame(model_0.groups_[0]['p_w_tw'], index=vec.get_feature_names())
# topics_model0_l2 = pd.DataFrame(model_0.groups_[2]['p_w_tw'], index=vec.get_feature_names())

# plot
model_1.state_.draw(layout='bipartite', subsample_edges=1000, hshortcuts=1, hide=0, hvertex_size=5, 
    output = '/mnt/h/topsbm_preprocessing/tweets_model_1.pdf')
    
# get number of words and docs
words_m1 = model_1.n_features_ # words : 43429
docs_m1 = model_1.n_samples_ # documents : 10000

# most preprocessing ===========================================================

texts_2 = preprocess(tweets_list, to_lower=True, rm_stops=True, rm_punct=True, lemmatize=True)

model_2, vec_2 = est_model(texts_2)

# number of topic groups
len(model_2.groups_) # 1

# # get word probs per topic
topics_model2_l0 = pd.DataFrame(model_2.groups_[0]['p_w_tw'], index=vec_2.get_feature_names())

# plot
model_2.state_.draw(layout='bipartite', subsample_edges=1000, hshortcuts=1, hide=0, hvertex_size=5, 
    output = '/mnt/h/topsbm_preprocessing/tweets_model_2.pdf')
    
# get number of words and docs
words_m2 = model_1.n_features_ # words : 43429
docs_m1 = model_1.n_samples_ # documents : 10000

# get top 10 words for topics in level 0
top_ten_model_2_l0 = []
for topic in topics_model2_l0.columns:
    print(topics_model2_l0[topic].nlargest(10))
    top_ten_model_2_l0.append(topics_model2_l0[topic].nlargest(10))

# save
writer=pd.ExcelWriter('/mnt/h/topsbm_preprocessing/tweets_model2_l0.xlsx')
for i, A in enumerate(top_ten_model_0_l0):
    A.to_excel(writer,sheet_name=f"topic_{i}")
writer.save()
