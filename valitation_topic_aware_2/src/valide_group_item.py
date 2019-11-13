import networkx as nx
import pandas as pd
import random
import copy
import logging

def find_small_degree(G,core_x):
    #找到所有degree小于x-core的节点
    degrees = nx.degree(G)
    degrees = [list(i) for i in degrees]
    nodes_list = []
    for i in degrees:
        if i[1] < core_x:
            nodes_list.append(i[0])
    return nodes_list

def x_core_graph(G,core_x):
    #将巨大的Graph图中所有degree小于x-core的节点去掉，直到所有点的degree>=x-core
    nodes = G.nodes()
    print(len(nodes))

    nodes_list = find_small_degree(G, core_x)
    while nodes_list:
        G.remove_nodes_from(nodes_list)
        #nodes = G.nodes()
        #print(len(nodes))
        nodes_list = find_small_degree(G, core_x)
    return G

def choose_m_senders(G_x,groupsize_m,core_x):
    #不能从x-core 的graph中随机选择m个节点，需要一点一点去掉多余的点（每次去掉都会改变degree，重新找x-core子图）
    G = copy.deepcopy(G_x)
    nodes=G.nodes()
    #print('the begin nb=',len(nodes))
    while len(nodes())>groupsize_m :
        v = random.sample(nodes, 100)
        #print(v)
        G.remove_nodes_from(v)
        G = x_core_graph(G, core_x)
        nodes = G.nodes()
        if len(nodes)<=10*groupsize_m:
            break
    while len(nodes):
        if len(nodes)<=groupsize_m:
            break
        v = random.sample(nodes, 1)
        #print(v)
        G.remove_nodes_from(v)
        G = x_core_graph(G, core_x)
        nodes = G.nodes()
    return G

def initial_groups_to_items(links, ratings, core_x, k, groupsize_m):
    exist_movies = list(set(list(ratings['movieid'])))

    a = list(links['user_id'])
    b = list(links['friend_id'])
    # l1=list(set(a).union(set(b)))
    edges1 = list(zip(a, b))

    G = nx.Graph()
    G.add_edges_from(edges1)
    #print('len=',len(G.nodes()))
    G_x=x_core_graph(G,core_x)
    nodes =G_x.nodes()

    #从x-core 的graph 中随机挑选 m 个 senders,
    while True:
        print('select one more time')
        new_G=choose_m_senders(G_x, groupsize_m, core_x)
        nodes=new_G.nodes()
        if len(nodes)==groupsize_m:
            mSender_ratings = ratings[ratings['userid'].isin(nodes)]
            if len(mSender_ratings)>1:
                break

    #print('ok')
    #print(len(new_G.nodes()))
    #print(new_G.degree())

    senders=new_G.nodes()
    #print(mSender_ratings)

    mSender_ratings.to_csv('movie_rated_by_groupsizem.txt', sep='\t', index=False)
    # 将group——sizem所评论过的电影排序
    a=mSender_ratings['movieid'].value_counts()
    print(a)
    #排序后，找到前top k 个电影
    list_top_k = []
    for i, u in a.items():
        #print(i, u)
        list_top_k.append(i)
        if len(list_top_k) == k:#这里需要考虑到一个问题：可能遍历结束list_top_k的数量小于k,当mSender_ratings记录比较少时
            break
    #print(list_top_k)

    #随机选择 10*k 个其他电影
    rest_movies=list(set(exist_movies)-set(list_top_k))
    list_random = random.sample(rest_movies, 10*k)

    #前top k个和random出的10*k个电影一起作为list of items
    items=list_top_k+list_random
    print(items)

    logger.info('wait_items ={}'.format(items))
    logger.info('senders ={}'.format(senders))

    text_save([items], 'output/wait_info.txt')
    text_save([senders], 'output/wait_info.txt')

    #######################################Delete the evaluation relationship between item and senders##############################################
    # With the items to be evaluated and senders, delete the rating between the senders and items - log
    ratings_after_cut = ratings.drop(
        ratings.loc[(ratings['userid'].isin(senders)) & (ratings['movieid'].isin(items))].index)
    print(len(ratings))
    print(len(ratings_after_cut))
    ratings_after_cut.to_csv(str(k) + 'ratings_after_cut', sep='\t', index=False)
    return items,senders
