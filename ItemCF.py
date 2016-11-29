import math
from numpy import *
import datetime

NumOfItems=1690

def GetData(datafile='u.data'):
    '''
    把datafile文件中的数据读出来，返回data对象
    :param datafile: 数据源文件名称
    :return: 一个列表data，每一项是一个元组，元组第一项是用户，第二项是电影
    '''
    data=[]
    try:
        file = open(datafile)
    except:
        print("No such file named "+datafile)
        return
    for line in file:
        line=line.split('\t')
        try:
            data.append((int(line[0]),int(line[1])))
        except:
            pass
    file.close()
    return data

def SplitData(data,M,k,seed):
    '''
    train/data=1/M
    :param data: 传入的数据
    :param M: 测试集占比
    :param k: 一个任意的数字，用来随机筛选出测试集和训练集
    :param seed: 随机数种子
    :return: train:训练集  test:测试集 都是字典，key是用户id，value是电影集合
    '''
    test=dict()
    train=dict()
    random.seed(seed)
    #在M次实验里面我们需要相同的随机数种子，这样生成的随机序列是相同的
    for user, item in data:
        if random.randint(0,M)!=k:
            # 相等的概率是1/M，所以M决定了测试集在所有数据中的比例
            if user not in test.keys():
                test[user] = set()
            test[user].add(item)
        else:
            if user not in train.keys():
                train[user]=set()
            train[user].add(item)
    return train,test

def getW(train):
    #train本身已经是用户-物品倒排表
    #W[u][v]表示u和v物品的相似度
    W=zeros([NumOfItems,NumOfItems],dtype=float16)
    #C[u][v]表示喜欢u又喜欢v的用户有多少个
    C=zeros([NumOfItems,NumOfItems],dtype=float16)
    #C=zeros([NumOfItems,NumOfItems],dtype=int32)
    #N[u]表示有多少用户喜欢u
    N=zeros([NumOfItems],dtype=int32)

    item_relateditems=dict()


    for user,items in train.items():
        for item1 in items:
            N[item1]+=1
            for item2 in items:
                if item1==item2:
                    continue
                if item1 not in item_relateditems:
                    item_relateditems[item1]=set()
                item_relateditems[item1].add(item2)
                C[item1][item2]+=(1/math.log(1+len(items)*1.0))
                #C[item1][item2]+=1

    for item1 in range(1,NumOfItems):
        if item1 in item_relateditems:
            for item2 in item_relateditems[item1]:
                W[item1][item2]=C[item1][item2]/sqrt(N[item1]*N[item2])

    return W,item_relateditems


def k_similar_item(W,item_relateditems,k):
    '''
    返回一个字典，key是每个item，value是item对应的k个最相似的物品
    '''
    begin=datetime.datetime.now()

    k_similar=dict()
    for i in range(1,NumOfItems):
        relateditems=dict()
        try:
            for x in item_relateditems[i]:
                relateditems[x]=W[i][x]
            relateditems=sorted(relateditems.items(),key=lambda x:x[1],reverse=True)
            k_similar[i]=set(dict(relateditems[0:k]))
        except KeyError:
            print(i," doesn't have any relateditems")
            k_similar[i]=set()
            for x in range(1,k+1):
                k_similar[i].add(x)

    end=datetime.datetime.now()
    print("it takes ",(end-begin).seconds," seconds to get k_similar_item for all items.")
    return k_similar


def GetRecommendation(user,train,W,relateditems,k,N,k_similar_item):
    rank=dict() #key是电影id，value是兴趣大小
    
    for i in range(NumOfItems):
        rank[i]=0


    possible_recommend=set()
    for item in train[user]:
        possible_recommend=possible_recommend.union(relateditems[item])

    for item in possible_recommend:
        k_items=k_similar_item[item]
        # k_items=dict()
        # if item in relateditems:
        #     for item2 in relateditems[item]:
        #         if item2==item:
        #             continue
        #         k_items[item2]=W[item][item2]
        # k_items=sorted(k_items.items(),key=lambda x:x[1],reverse=True)[0:k]

        # k_items=dict(k_items)

        for i in k_items:
            if i in train[user]:
                rank[item]+=1.0*W[item][i]

    for rank_key in rank:
        if rank_key in train[user]:
            rank[rank_key]=0
    return dict(sorted(rank.items(),key=lambda x:x[1],reverse=True)[0:N])


def Recall(train, test, N,k):
    '''
    :param train: 训练集
    :param test: 测试集
    :param N: TopN推荐中的N数目
    :return: 返回召回率
    '''
    hit=0 #预测准确的数目
    total=0 #所有的行为总数
    W,relateditems=getW(train)
    for user in train.keys():
        tu=test[user]
        rank=GetRecommendation(user,train,W,relateditems,k,N)
        for item in rank:
            if item in tu:
                hit+=1
        total+=len(tu)
    return hit/(total*1.0) #把结果转化成小数

def Precision(train,test,N,k):
    '''
    :param train: 训练集
    :param test: 测试集
    :param N: topN推荐中的数目N
    :return: 返回准确率
    '''
    hit=0
    total=0
    W,relateditems=getW(train)
    for user in train.keys():
        tu=test[user]
        rank=GetRecommendation(user, train,W,relateditems,k,N)
        for item in rank:
            if item in tu:
                hit+=1
        total+=N
    return hit/(total*1.0)

def Coverage(train,test,N,k):
    '''
    计算覆盖率
    :param train: 训练集，字典user->items
    :param test: 测试集,字典user->items
    :param N: topN推荐中的N
    :return: 覆盖率
    '''
    recommend_items=set()
    all_items=set()
    W,relateditems = getW(train)
    for user in train.keys():
        for item in train[user]:
            all_items.add(item)
        rank=GetRecommendation(user, train,W,relateditems,k,N)
        for item in rank:
            recommend_items.add(item)
    return len(recommend_items)/(len(all_items)*1.0)

def Popularity(train,test,N,k):
    '''
    计算新颖度
    :param train: 训练集,字典user->items
    :param test: 测试集,字典user->items
    :param N: topN推荐中的推荐数目N
    :return: 新颖度
    '''
    item_popularity=dict()
    W,relateditems=getW(train)
    for user, items in train.items():
        for item in items:
            if item not in item_popularity:
                item_popularity[item]=0
            item_popularity[item]+=1
    ret = 0
    n = 0
    for user in train.keys():
        rank = GetRecommendation(user, train, W,relateditems, k,N)
        for item in rank:
            ret += math.log(1 + item_popularity[item])
            n += 1
    ret /= n * 1.0
    return ret

def evaluate(train,test,N,k):
    recommends=dict()
    W,relateditems=getW(train)
    k_similar=k_similar_item(W,relateditems,k)
    for user in test:
        recommends[user]=GetRecommendation(user,train,W,relateditems,k,N,k_similar)


    recall=0
    hit=0 #预测准确的数目
    total=0 #所有的行为总数
    for user in train.keys():
        tu=test[user]
        rank=recommends[user]
        for item in rank:
            if item in tu:
                hit+=1
        total+=len(tu)
    recall=hit/(total*1.0) #把结果转化成小数

    precision=0
    hit=0 #预测准确的数目
    total=0 #所有的行为总数
    for user in train.keys():
        tu=test[user]
        rank=recommends[user]
        for item in rank:
            if item in tu:
                hit+=1
        total+=N
    precision=hit/(total*1.0) #把结果转化成小数

    coverage=0
    recommend_items=set()
    all_items=set()
    for user in train.keys():
        for item in train[user]:
            all_items.add(item)
        rank=recommends[user]
        for item in rank:
            recommend_items.add(item)
    coverage=len(recommend_items)/(len(all_items)*1.0)

    Popularity=0
    item_popularity=dict()
    for user, items in train.items():
        for item in items:
            if item not in item_popularity:
                item_popularity[item]=0
            item_popularity[item]+=1
    ret = 0
    n = 0
    for user in train.keys():
        rank = recommends[user]
        for item in rank:
            ret += math.log(1 + item_popularity[item])
            n += 1
    ret /= n * 1.0
    Popularity=ret

    return recall,precision,coverage,Popularity



def test1():
    data=GetData()
    train,test=SplitData(data,2,1,1)
    del data
    user=int(input("input the user id \n"))
    print("the train set contains the movies of the user: \n")
    print(train[user])
    N=int(input("input the number of recommendations\n"))
    k=int(input("input the number of related items\n"))
    starttime=datetime.datetime.now()
    W,relateditems=getW(train)
    
    endtime=datetime.datetime.now()
    print("it takes ",(endtime-starttime).seconds," seconds to get W")

    k_similar=k_similar_item(W,relateditems,k)

    starttime=datetime.datetime.now()
    recommend=GetRecommendation(user,train,W,relateditems,k,N,k_similar)
    endtime=datetime.datetime.now()
    print("it takes ",(endtime-starttime).seconds," seconds to get recommend for one user")
    print(recommend)
    for item in recommend:
        print(item),
        if(item in test[user]):
            print("   True")
        else:
            print("   False")



def test2():
    N=int(input("input the number of recommendations: \n"))
    k=int(input("input the number of related items: \n"))
    data=GetData()
    train,test=SplitData(data,2,1,1)
    del data
    recall,precision,coverage,popularity=evaluate(train,test,N,k)
    print("Recall: ",recall)
    print("Precision: ",precision)
    print("Coverage: ",coverage)
    print("Popularity: ",popularity)

if __name__=='__main__':
    test2()