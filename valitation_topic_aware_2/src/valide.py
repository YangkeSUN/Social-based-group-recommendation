# -*- coding: UTF-8 -*-
import json
import copy
import random
import argparse
import logging
from collections import Counter
from multiprocessing import Pool
#from functools import reduce

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from logzero import setup_logger
from sklearn import preprocessing

import valide_math as formula
import valide_graph as ns_graph
import valide_greedy as ns_greedy
import valide_util as ns_util
import valide_item_group as ns_item_group
import valide_group_item as ns_group_item

logger=setup_logger(name='mylogger',logfile='main.log',level=logging.INFO)

#def generate_PR_relevence(G, dict_gamma): double_edge_label logger.info
#def userid_group_json(rating_filename): pd.read_csv 
#def nodelta_prop(links_filename,genres,infogroup):
#def chose_item_sender(inter_info,ratings,k):
#def choose_gamma(m,k_size,wait_items,genres):
#def initial(k):
#def run_one_time(config, k,coef_sw,coef_f):
#def validation(config, nb_item,coef_sw,coef_f):
#def load_config(config_path):
#def main():

def generate_PR_relevence(G, dict_gamma):
    '''
    :param G:a network associated weight
    :param dict_gamma: every item's vector gamma
    :return: every user's personal interests ：relevence(u,i)=relevence(u,t)*gamma_i
    '''
    V = list(G.nodes)
    edge_labels = ns_graph.double_edge_label(G)
    rel_user_item = {}
    for v in V:
        C = G[v]  # find node v's all neigbors , type: dict
        list_pp = np.array([edge_labels[(v, i)] for i in C.keys()])
        max_pp = list_pp.max(axis=0)
        rel_u_t = [random.uniform(pp, 0.99) for pp in max_pp]
        rel = []
        for key, gamma_i in dict_gamma.items():
            rel.append(np.dot(rel_u_t, dict_gamma[key]))
        rel_user_item[str(v)] = list(rel)
    #logger.info("rel_user_item={}".format(rel_user_item))
    return rel_user_item
#end def generate_PR_relevence 


###################################关于计算pr##########################################
def userid_group_json(k_ratings):
    d = dict()
    for _, row in k_ratings.iterrows():
        user_id, movie_id, rating,time = row
        d.setdefault(user_id, {}).update({movie_id: [rating,time]})
    #ns_util.smart_json_write(d, 'user_infogroup.json.gz')
    return d
#end def userid_group_json

def nodelta_prop(links, genres, infogroup):
    prob = {}
    useful_edges_info={}
    for (u, v) in links.itertuples(index=False):
        try:
            info_u = infogroup[u]
        except:
            # logger.info('user u={} didnt rate any movie'.format(u))
            #useless_edges.append(edge)
            continue
        try:
            info_v = infogroup[v]
        except:
            #useless_edges.append(edge)
            #logger.info('user v={} didnt rate any movie'.format(v))
            continue
        movie_u = list(info_u.keys())
        movie_v = list(info_v.keys())
        # 两个用户评论电影的并集
        total = list(set(movie_u).union(set(movie_v)))
        # logger.info('all movies as denominator={}'.format(total))
        n1 = len(movie_u)
        n2 = len(movie_v)
        # print('the denominator is ',n)
        # 两个用户评论电影的交集
        movie_for_u_v = list(set(movie_u).intersection(set(movie_v)))
        # logger.info('the intersection between u and v is={}'.format(movie_for_u_v))
        if len(movie_for_u_v) == 0:
            #useless_edges.append(edge)
            continue
        list_id_u_v = []
        list_id_v_u = []
        for i in movie_for_u_v:
            # print('we see the movie:',i)
            rating_u = info_u[i][0]
            rating_v = info_v[i][0]
            time_u = info_u[i][1]
            time_v = info_v[i][1]
            #delta_u_v = abs(rating_u - rating_v)
            if rating_u <= rating_v and time_u <= time_v:
                list_id_u_v.append(i)
            elif rating_v <= rating_u and time_v <= time_u:
                list_id_v_u.append(i)
        #logger.info('we begin to translate topic')
        sum1 = np.zeros(29)
        for movie in list_id_u_v:
            try:
                data1 = np.array(genres[movie])
            except:
                # logger.info('we dont have this movie={}'.format(movie))
                continue
            sum1 = sum1 + data1

        prob1 = sum1 / n1
        sum2 = np.zeros(29)
        for movie2 in list_id_v_u:
            try:
                data2 = np.array(genres[movie2])
            except:
                # logger.info('we dont have this movie={}'.format(movie2))
                continue
            sum2 = sum2 + data2
        prob2 = sum2 / n2
        useful_edges_info[str((u, v))]=movie_for_u_v
        prob[str((u, v))] = list(prob1)
        prob[str((v, u))] = list(prob2)

    ns_util.smart_json_write(prob, 'prob_nodelta.json')
    ns_util.smart_json_write(useful_edges_info, '_interaction_info.json')
    return prob
#end def nodelta_prop

def top_n_most_seen_film(ratings, n):
    return Counter(list(ratings['movieid'])).most_common(n)

def top_n_most_seen_film_within_group(group, ratings, n):
    return Counter(map(lambda x: x['movieid'], filter((lambda x: x['userid'] in group), ratings.itertuples(index=False)))).most_common(n)

def chose_item_sender(links, ratings, nb_item, groupsize, x_core, top_n):
    #items, senders = ns_group_item.initial_groups_to_items(links, ratings, x_core, nb_item, groupsize)
    items, senders = ns_item_group.initial_items_to_groups(ratings, top_n, nb_item, groupsize)
    return items,senders
#end def chose_item_sender


def choose_gamma(m,nb_item,wait_items,genres):
    #挑选m个items，并与gamma做成一一对应的字典，其中包含带检测的wait_items
    item_gamma = {}
    records = {}

    # 随机选取m-k个item
    item_gamma0 = random.sample(genres.items(), m - nb_item)

    # 将item与gamma一一对应做成字典
    for i, (movie, g) in enumerate(item_gamma0):
        item_gamma[i] = np.array(g) / sum(np.array(g))
        records[i] = movie
    #将wait_item加入到字典中
    for i in range(m - nb_item, m):
        records[i] = wait_items[i % nb_item]
    new_records = {value: key for key, value in records.items()}

    for item in wait_items:
        index = new_records[item]
        item_gamma[index] = np.array(genres[str(item)]) / sum(np.array(genres[str(item)]))
    '''
    #将字典转化为dataframe并保存在input下
    df = pd.DataFrame.from_dict(records, orient='index')
    df['item_id'] = df.index
    df = df.rename(columns={0: 'movie_id'})
    df1 = df[['item_id', 'movie_id']]
    # print(df1)
    item_gamma2={key:str(list(value)) for key,value in item_gamma.items()}
    dff = pd.DataFrame.from_dict(item_gamma2, orient='index')
    dff['item_id'] = dff.index
    dff = dff.rename(columns={0: 'vec_genres'})
    dff1 = dff[['item_id', 'vec_genres']]

    DF=pd.merge(df1,dff1)
    DF.to_csv('output/Records_movie_id'+str(nb_item), sep='\t', index=False)
    '''
    return item_gamma
#end def choose_gamma

def initial(genres, links, ratings, top_n, groupsize, x_core, nb_item):
    # 做一些准备工作，得到待观察的items以及sender group，并记录号cut掉关系的ratings

    # print('all logs=',len(ratings))
    # print('all items=',len(set(list(ratings['movieid']))))

    exist_items = list(genres.keys())
    # print(len(exist_items))

    ####################################### 选择 k 个 待观察的items，以及 senders#################################
    items, users = chose_item_sender(links, ratings, nb_item, groupsize, x_core, top_n)

    while len(items) != nb_item or len(users) < groupsize:
        print('initial: not ok')
        items, users = chose_item_sender(links, ratings, nb_item, groupsize, x_core, top_n)

    senders = random.sample(users, groupsize)

    # logger.info('wait_items ={}'.format(items))
    # logger.info('senders ={}'.format(senders))
    ns_util.text_save([items], '../output/wait_info.txt')
    ns_util.text_save([senders], '../output/wait_info.txt')

    #######################################删除item与senders间的评价关系##########################################
    # 有了待评价的items以及senders之后，删除senders和items之间的rating——log。
    ratings_after_cut = ratings.drop(
        ratings.loc[(ratings['userid'].isin(senders)) & (ratings['movieid'].isin(items))].index)
    # print(len(ratings))
    # print(len(ratings_after_cut))
    # ratings_after_cut.to_csv(str(nb_item)+'ratings_after_cut', sep='\t', index=False)

    return items, senders


# end def initial


def run_one_time(config, genres, links, ratings, top_n, G, nb_item, groupsize, x_core, coef_sw, coef_f):
    # step1：确定观察对象个数k-size 为k

    # step1：确定观察对象个数k-size 为k

    # step2：选择k个观察对象与senders，并切断他们之间的rating记录
    wait_items, list_S = ns_item_group.initial_items_to_groups(ratings, top_n, nb_item, groupsize)

    # 233行 改为：
    # wait_items, list_S = ns_group_item.initial_groups_to_items(links, ratings, x_core, nb_item, groupsize)
    # wait_items, list_S = ns_item_group.initial_items_to_groups(ratings, top_n, nb_item, groupsize)

    # step3：根据cut后的rating以及links信息，计算社交图里，每条边的传播概率（topic-aware），生成prob的json文件储存相关信息
    # k_rating_after_cut_filepath = str(nb_item) + 'ratings_after_cut'
    # rnames = ['user_id', 'movie_id', 'rating', 'time']
    # k_ratings = pd.read_csv(k_rating_after_cut_filepath, skiprows=[0], sep='\t', header=None, names=rnames,engine='python')
    # infogroup = userid_group_json(k_ratings)

    ###################计算出pr的值，写入prob_nodelta,json##################################################
    #nodelta_prop(links, genres, infogroup)

    # 准备greedy算法的input
    n = len(list_S)
    m = 10*len(list_S)
    #####################################建立social network 社交图###########################################
    #print('G is ok')

    ###################################挑选m 个 item 及其对应的 gamma,组成字典##################################
    item_gamma = choose_gamma(m, nb_item, wait_items, genres)
    #print("gamma for every item is  ", item_gamma)

    ##########################得到rel（u,i）##############################################################
    PR_rel = generate_PR_relevence(G, item_gamma)
    # print("PR_rel=", PR_rel)
    #print('rel is ok')

    ###########################初始化####################################################
    nb_topic = 29
    User, Item, X = ns_graph.init_reset(n, m)
    #print("User=", User)
    #print("Item=", Item)
    #print("X=", X)
    gamma = np.zeros(nb_topic)
    reco, benefit = ns_greedy.greedy(G, list_S, gamma, item_gamma, Item, nb_item, PR_rel, X, coef_sw, coef_f)
    #print("our recommendations' list is :", reco)
    #print("there are ", len(reco), "items")
    print(benefit)
    x = [j for j in range(1, nb_item + 1)]
    y = benefit
    return benefit
#end def run_one_time

#def create_k_core_group(links):


def validation(config, nb_item_range, groupsize, x_core, coef_sw, coef_f, top_range):
    # step1：确定观察对象个数k-size
    genres = ns_util.smart_json_load(config['input']['genres'])
    prob = ns_util.smart_json_load(config['input']['prob_nodelta'])
    #inter_info = ns_util.smart_json_load(config['input']['useful_edges_interaction_info'])
    mnames = ['user_id', 'friend_id']  # genres means tags
    links = pd.read_csv(config['input']['links_after_2wash'], skiprows=[0], sep='\t', header=None, names=mnames, engine='python')
    mnames = ['userid', 'movieid', 'rating', 'date']
    ratings = pd.read_csv(config['input']['ratings_after_3wash'], skiprows=[0], sep='\t', header=None, names=mnames, engine='python')
    top_n = top_n_most_seen_film(ratings, top_range)
    top_n1 = list(map(lambda x: x[0], top_n))
    G = ns_graph.topic_network(links, prob, nb_topic=29)
    print('files loaded')

    precision_validation = []
    for nb_item in range(1, nb_item_range):
        items = [i for i in range(groupsize - nb_item, groupsize)]
        recommendation = run_one_time(config, genres, links, ratings, top_n1, G, nb_item, groupsize, x_core, coef_sw, coef_f)
        #valide = reduce((lambda count, x: count+1), filter(lambda x: x in item), recomendation)
        valide = 0
        for item in recommendation:
            if item in items:
                valide = valide + 1
        #print(valide)
        p = valide / nb_item
        precision_validation.append(p)
    return precision_validation
#end def validation

def load_config(config_path):
    with ns_util.smart_open(config_path) as config_file:
        return json.load(config_file)
#end def load_config

def main():
    parser = argparse.ArgumentParser(description='recommend films to groups of user')
    parser.add_argument('--config', nargs=1, metavar='FILE', type=str, required=False, default='../config.json'
            , help='configuration file containing input file locations and default parameters (default=config.json)')
    parser.add_argument('--nb_item_range', nargs=1, metavar='K', type=float, required=False, help='number of items_to_be_tested')
    parser.add_argument('--coef_sw', nargs=1, metavar='A', type=float, required=False, help='Social welfare coeficient (between 0.0 and 1.0)')
    parser.add_argument('--coef_f', nargs=1, metavar='B', type=float, required=False, help='faireness coeficient (between 0.0 and 1.0)')
    parser.add_argument('--groupsize', nargs=1, metavar='M', type=float, required=False, help="number of users in a group")
    parser.add_argument('--x_core', nargs=1, metavar='X', type=float, required=False, help="minim1um number of links from one user to other users in a group")
    parser.add_argument('--top_range', nargs=1, metavar='T', type=float, required=False, help="numbers of top item that we choose our target items")
    args = parser.parse_args()
    
    config = load_config(args.config)
    if (args.coef_sw != None):
        config["coef_sw"] = args.coef_sw[0]
    if (args.coef_f != None):
        config["coef_f"] = args.coef_f[0]
    if (args.groupsize != None):
        config["groupsize"] = args.groupsize[0]
    if (args.x_core != None):
        config["x_core"] = args.x_core[0]
    if (args.nb_item_range != None):
        config["nb_item_range"] = args.nb_item_range[0]
    if (args.top_range != None):
        config["top_range"] = args.top_range[0]
    precision_validation = validation(config, config["nb_item_range"], config["groupsize"], config["x_core"], config["coef_sw"], config["coef_f"], config["top_range"])
    ns_util.smart_json_write(precision_validation, config["output_file"])
#end def main

if __name__ == '__main__':
    main()
