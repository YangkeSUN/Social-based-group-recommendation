import networkx as nx
import pandas as pd
import random
import copy
import logging

def item_to_group(ratings,top_M_items,nb_item):

    #randomly choose k items from the Top M
    list_items = random.sample(top_M_items, nb_item)
    #print(list_items)

    #根据之前的方法选择出所有评论过k个item的用户
    df = ratings[ratings['movieid'].isin(list_items)]
    #print(df)
    senders=[]
    groups=df.groupby('userid')
    for name,group in groups:
        #print(group)
        movies=group['movieid']
        #print(list(movies))
        if len(movies)==nb_item:
            #print(len(movies))
            senders.append(name)
    #print('senders=',senders)
    items=list(set(list(df['movieid'])))
    #print('items=',items)
    #print('senders=', senders)
    return items,senders


def initial_items_to_groups(ratings, top_M_items, nb_item, groupsize_m):
    ####################################### Select k items to be observed, and m senders #################################
    items, users = item_to_group(ratings, top_M_items, nb_item)

    count = 0
    while ((len(items) != nb_item) or (len(users) < groupsize_m)):
        print('initial_items_to_groups: not ok, count: ' + str(count) + ', len items: ' + str(len(items)) + ', len users: ' + str(len(users)))
        count += 1
        items, users = item_to_group(ratings, top_M_items, nb_item)

    senders = random.sample(users, groupsize_m)
    #logger.info('wait_items ={}'.format(items))
    #logger.info('senders ={}'.format(senders))
    #text_save([items], 'output/wait_info.txt')
    #text_save([senders], 'output/wait_info.txt')

    #######################################Delete the evaluation relationship between item and senders##############################################
    # With the items to be evaluated and senders, delete the rating between the senders and items - log
    ratings_after_cut = ratings.drop(
        ratings.loc[(ratings['userid'].isin(senders)) & (ratings['movieid'].isin(items))].index)
    #print(len(ratings))
    #print(len(ratings_after_cut))
    ratings_after_cut.to_csv('../output/' + str(nb_item) + 'ratings_after_cut', sep='\t', index=False)

    return items, senders
