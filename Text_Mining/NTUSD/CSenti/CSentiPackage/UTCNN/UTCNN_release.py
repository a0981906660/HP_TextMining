#!/usr/bin/env python
# -*- coding: utf-8 -*-

# user topic comment neural network
import numpy as np 
import sys,time,pickle,ConfigParser

from keras.models import Model
from keras.layers import Embedding, Input, merge
from keras.layers.core import Lambda, Reshape, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import precision_recall_fscore_support

def loadEmbedding(Embfile,dim):
    print 'Load embedding file:', Embfile
    voc = {}
    try:
        with open(Embfile) as f:
            lines = [line.rstrip('\n') for line in f]
    except:
        print 'Input error',Embfile
    emb=np.zeros((len(lines),dim))    # note: add a "zero" embedding for zero padding
    for l, line in enumerate(lines):
        tokens = line.split(' ')
        emb[l]= [float(t) for t in tokens[1:]]
        voc[tokens[0]] = l+2
    average = np.average(emb, axis=0)
    zero = np.zeros(dim)
    emb = np.append([zero,average],emb,axis=0)
    voc['<zero>'] = 0
    voc['<unk>'] = 1    
    return voc,emb

def loadTrainData(dataset):
    print 'Load data file:', dataset
    try:
        with open(dataset) as f:
            lines = [line.rstrip('\n') for line in f]
    except:
        print 'Input error',dataset
    uu=[]   #users, authors and likers' ids
    tt=[]   #topics
    ll=[]   #labels
    cc=[]   #contents
    rr=[]   #commenters
    mm=[]   #comments
    for line in lines:
        tokens = line.split('\t\t')
        uu.append(tokens[0])
        tt.append(tokens[1])
        ll.append(tokens[2])
        cc.append(tokens[3])
        rr.append(tokens[4])
        mm.append(tokens[5])
        
    ## process users and commenters
    ulist=[]
    rlist=[]
    users=[]
    
    for u in uu:
        ulist.append(np.array(u.strip().split(' ')))
        users.extend(u.strip().split(' '))
    for r in rr:
        rlist.append(np.array(r.strip().split(' ')))
        users.extend(r.strip().split(' '))
    uset=list(set(users))

    uDic={}
    uDic['<zero>'] = 0
    for i,u in enumerate(uset):
        uDic[u]=i+1
    
    ## procsee topics
    tlist=[]
    topics=[]
    for t in tt:
        tlist.append(np.array(t.strip().split(' ')))
        topics.extend(t.strip().split(' '))
    tset = list(set(topics))
    tDic={}
    tDic['<zero>'] = 0
    for i,t in enumerate(tset):
        tDic[t]=i+1
        
    ## process label
    llist = np.array([int(l)+1 for l in ll])
    
    # process content
    clist =[]
    for i in range(len(cc)):
        sentences = cc[i].split('<sssss>')
        clist.append([s.rstrip().split(' ') for s in sentences])
    
    # process comment
    mlist =[]
    for i in range(len(mm)):
        comments = mm[i].split(',')
        mlist.append([m.strip().split(' ') for m in comments])
    
    # check comments
    for i in range(len(mlist)):
        mLen = len(mlist[i])
        rLen = len(rlist[i])
        if mLen <> rLen:
            print 'Input error'
            print '#comments != #commenters'
            print mLen ,'!=', rLen
            print 'in document', i
            print 'commenter:',rlist[i]
            print 'comment'
            for im,m in enumerate(mlist[i]):
                print 'comment',im,':',' '.join([mm.decode('utf8') for mm in m])
            sys.exit()
    
    return ulist, uDic, tlist, tDic, llist, clist, mlist, rlist

def contains(subset, sets):
    newset=[]
    for s in subset:
        if s in sets:
            newset.append(s)
    return newset

def contains_cmtr(comments,commenters,uDic):
    new_comments =[]
    new_commenters=[]
    for i,r in enumerate(commenters):
        if r in uDic:
            new_commenters.append(r)
            new_comments.append(comments[i])
    return new_comments,new_commenters

def loadTestData(dataset,uDic,tDic):
    print 'Load data file:', dataset
    try:
        with open(dataset) as f:
            lines = [line.rstrip('\n') for line in f]
    except:
        print 'Input error',dataset
    uu=[]   #users, authors and likers' ids
    tt=[]   #topics
    ll=[]   #labels
    cc=[]   #contents
    rr=[]   #commenters
    mm=[]   #comments
    for line in lines:
        tokens = line.split('\t\t')
        users = np.array(tokens[0].strip().split(' '))
        new_users = contains(users,uDic)
        
        commenters = np.array(tokens[4].strip().split(' '))
        comments = tokens[5].split(',')
        new_comments, new_commenters = contains_cmtr(comments,commenters,uDic)
        
        topics = np.array(tokens[1].strip().split(' '))
        new_topics = contains(topics,tDic)
        if len(new_users)>0 and len(new_topics)>0: 
            uu.append(new_users)
            tt.append(new_topics)
            ll.append(tokens[2])
            cc.append(tokens[3])
            mm.append(new_comments)
            rr.append(new_commenters)
    
    ulist = uu
    tlist = tt
    llist = np.array([int(l)+1 for l in ll])
    
    clist =[]
    for i in range(len(cc)):
        sentences = cc[i].split('<sssss>')
        clist.append([s.rstrip().split(' ') for s in sentences])
    
    mlist =[]
    for i in range(len(mm)):
        mlist.append([m.strip().split(' ') for m in mm[i]])
    
    rlist =rr
    
    for i in range(len(mlist)):
        mLen = len(mlist[i])
        rLen = len(rlist[i])
        if mLen <> rLen:
            print '#comments != #commenters'
            print 'mLen != rLen'
            sys.exit()
        
    return ulist, tlist, llist, clist, mlist, rlist

def getMaxDoc(clists):
    dmax = 0
    smax = 0
    tmax = 0
    for clist in clists:
        for c in clist:
            t_temp = 0
            if len(c)>dmax:
                dmax = len(c)
            for cc in c:
                if len(cc)>smax:
                    smax = len(cc)
                t_temp += len(cc)
            if tmax<t_temp:
                tmax = t_temp
    
    return dmax, smax, tmax

def getMaxUser(ulists):
    umax = 0
    for ulist in ulists:
        for u in ulist:
            if umax<len(u):
                umax = len(u)
    return umax

def getMaxCmt(mlists):
    mmax = 0
    mLength =0
    for mlist in mlists:
        for m in mlist:
            if mmax<len(m):
                mmax = len(m)
            for mm in m:
                if mLength<len(mm):
                    mLength = len(mm)
    return mmax,mLength

def dicLookUp(clist,voc,samples,tLength):
    cArray = np.zeros((samples,tLength),dtype='int32')
    for ic,c in enumerate(clist):
        idx = 0
        for s in c:
            for t in s:
                if t in voc:
                    cArray[ic][idx] = voc[t]
                else:
                    cArray[ic][idx] = 1  ## unknown word
                idx = idx +1
    return cArray

def dicLookUpCmt(mlist,voc,samples,maxComment,maxCmtLength):
    cArray = np.zeros((samples,maxComment*maxCmtLength),dtype='int32')
    for im,m in enumerate(mlist):
        for imm,mm in enumerate(m):
            for it,t in enumerate(mm):
                if t in voc:
                    cArray[im][imm*maxComment+it] = voc[t]
                else:
                    cArray[im][imm*maxComment+it] = 1  ## unknown word
    return cArray

def getCmtWrapper(i):
    def getCmt(x):
        return x[:,i,:,:]
    return getCmt

def getCmtrWrapper(i):
    def getCmtr(x):
        return x[:,i,:]
    return getCmtr

def getCmtWrapper_output_shape(input_shape):
    shape = list(input_shape)
    return [shape[0]]+shape[2:]

def paddingUser(ulist,uDic,samples,maxUser):
    userlist = np.zeros((samples,maxUser),dtype='int32') 
    for iu,u in enumerate(ulist):
        for iuu,uu in enumerate(u):
            if uu in uDic:
                userlist[iu][iuu]=uDic[uu]
            else:
                userlist[iu][iuu]=0
    return userlist

def predictFromClass(classes):
    return np.argmax(classes,axis=1)

def findTrues(rlist,prediction):
    r = np.zeros((len(rlist),))
    for i in range(len(rlist)):
        if rlist[i] == prediction[i]:
            r[i] = 1
        else:
            r[i] = 0
    return r

def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

def readConfigs(configName):
    config = ConfigParser.ConfigParser()
    config.read(configName)
    sections = config.sections()
    dicFile = {}
    dicPar = {}
    for section in sections:
        options = config.options(section)
        for option in options:
            try:
                if section == 'Files':
                    dicFile[option] = config.get(section, option)
                else:
                    dicPar[option] = num(config.get(section, option))
            except:
                print 'Config syntax error:',sections,option
                sys.exit()
    return dicFile,dicPar

def tryAssign(dic,key):
    try:
        return(dic[key])
    except:
        print 'config error:',key
        sys.exit()

if __name__ == '__main__':
    
    argLen = len(sys.argv)
    if argLen<>2:
        print 'Error!'
        print 'Input example: python UTCNN_release.py config.ini'
        sys.exit()
    else:
        [dicFile,dicPar] = readConfigs(str(sys.argv[1]))
    t1 = time.time()
    embeddingFile = dicFile['embedding_file']
    vDim = tryAssign(dicPar,'v_dim')
    voc,embeddings = loadEmbedding(embeddingFile,vDim)
    print 'Load Embedding Elapse: ', time.time()-t1
    
    vSize = len(embeddings)
    t2 = time.time()
    trainFile = dicFile['train_file']
    testFile = dicFile['test_file']
    devFile = dicFile['dev_file']
        
    [ulistTrain,uDic, tlistTrain, tDic, llistTrain, clistTrain, mlistTrain, rlistTrain] = loadTrainData(trainFile)
    [ulistTest, tlistTest, llistTest, clistTest, mlistTest, rlistTest] = loadTestData(testFile,uDic,tDic)
    [ulistDev, tlistDev, llistDev, clistDev, mlistDev, rlistDev] = loadTestData(devFile,uDic,tDic)

    
    t3 = time.time()
    print 'Load Data Elapse: ', t3-t2

    uSize = len(uDic)
    uDim = tryAssign(dicPar,'u_dim')
    mini_uDim = tryAssign(dicPar,'mini_u_dim')
    
    tSize = len(tDic)
    tDim = tryAssign(dicPar,'t_dim')
    mini_tDim = tryAssign(dicPar,'mini_t_dim')
    
    conSize = tryAssign(dicPar,'con_size')
    
    lSize = tryAssign(dicPar,'l_size')
    
    samples = len(ulistTrain)
    samplesTest = len(ulistTest)
    samplesDev = len(ulistDev)
    
    [maxDocLength,maxSentenceLength,maxTotalLength] = getMaxDoc([clistTrain,clistTest,clistDev]) # number of sentence per document,# number of words per sentence
    [maxComment,maxCmtLength] = getMaxCmt([mlistTrain,mlistTest,mlistDev])
    maxUser = getMaxUser([ulistTrain,ulistTest,ulistDev])
    maxTopic = tryAssign(dicPar,'max_topic')
    
    print 'Train Max: ',getMaxDoc([clistTrain])
    print 'Test Max: ',getMaxDoc([clistTest])
    print 'Dev Max: ',getMaxDoc([clistDev])
    print 'max user:',maxUser
    print 'max comment:',maxComment
    print 'max comment length:',maxCmtLength
    print 'user length:',len(uDic)
    print 'topic length:',len(tDic)
    
    # padding users
    ulistTrain = paddingUser(ulistTrain,uDic,samples,maxUser)
    ulistTest = paddingUser(ulistTest,uDic,samplesTest,maxUser)
    ulistDev = paddingUser(ulistDev,uDic,samplesDev,maxUser)
    
    # padding topics
    tlistTrain = paddingUser(tlistTrain,tDic,samples,maxTopic)
    tlistTest = paddingUser(tlistTest,tDic,samplesTest,maxTopic)
    tlistDev = paddingUser(tlistDev,tDic,samplesDev,maxTopic)
    
    # padding commenters
    rlistTrain = paddingUser(rlistTrain,uDic,samples,maxComment)
    rlistTest = paddingUser(rlistTest,uDic,samplesTest,maxComment)
    rlistDev = paddingUser(rlistDev,uDic,samplesDev,maxComment)
         
    flength1 = tryAssign(dicPar,'flength1')
    flength2 = tryAssign(dicPar,'flength2')
    flength3 = tryAssign(dicPar,'flength3')
     
    rndBase = tryAssign(dicPar,'rnd_base')
 
    inputT1 = ulistTrain
    inputT2 = tlistTrain  
    inputT3 = dicLookUp(clistTrain,voc,samples,maxTotalLength)
    inputT4 = dicLookUpCmt(mlistTrain,voc,samples,maxComment,maxCmtLength)
    inputT5 = rlistTrain
     
    inputT1Test = ulistTest
    inputT2Test = tlistTest 
    inputT3Test = dicLookUp(clistTest,voc,samplesTest,maxTotalLength)  
    inputT4Test = dicLookUpCmt(mlistTest,voc,samplesTest,maxComment,maxCmtLength)
    inputT5Test = rlistTest
     
    inputT1Dev = ulistDev
    inputT2Dev = tlistDev  
    inputT3Dev = dicLookUp(clistDev,voc,samplesDev,maxTotalLength)
    inputT4Dev = dicLookUpCmt(mlistDev,voc,samplesDev,maxComment,maxCmtLength)
    inputT5Dev = rlistDev 
     
    inputU = Input(shape=(maxUser,),dtype='int32',name='inputU')                #users
    inputT = Input(shape=(maxTopic,),dtype='int32',name='inputP')               #topics
    inputC = Input(shape=(maxTotalLength,),dtype='int32',name='inputC')         #content
    inputM = Input(shape=(maxComment*maxCmtLength,),dtype='int32',name='inputM')#comments
    inputR = Input(shape=(maxComment,),dtype='int32',name='inputR')             #commenters

    rEUV = np.random.uniform(low=-rndBase,high=rndBase,size=(uSize,uDim))
    EUV = Embedding(output_dim=uDim,input_dim=uSize,input_length=maxUser,weights=[rEUV])
    embeddingU_v = EUV(inputU)
    eUV_reshape = Reshape((1,maxUser,uDim))(embeddingU_v)
    eUV_pooling = MaxPooling2D(pool_size=(maxUser,1),border_mode='valid')(eUV_reshape)
     
    rEUM = np.random.uniform(low=-rndBase,high=rndBase,size=(uSize,vDim*mini_uDim))
    EUM = Embedding(output_dim=vDim*mini_uDim,input_dim=uSize,input_length=maxUser,weights=[rEUM])
    embeddingU_m = EUM(inputU)
    eUM_reshape = Reshape((1,maxUser,vDim*mini_uDim))(embeddingU_m)
    eUM_pooling = MaxPooling2D(pool_size=(maxUser,1),border_mode='valid')(eUM_reshape)
     
    rETV = np.random.uniform(low=-rndBase,high=rndBase,size=(tSize,tDim))
    embeddingT_v = Embedding(output_dim=tDim,input_dim=tSize,input_length=maxTopic,weights=[rETV])(inputT)
    eTV_reshape = Reshape((1,maxTopic,tDim))(embeddingT_v)
    eTV_pooling = MaxPooling2D(pool_size=(maxTopic,1),border_mode='valid')(eTV_reshape)
     
    rETM = np.random.uniform(low=-rndBase,high=rndBase,size=(tSize,vDim*mini_tDim))
    embeddingT_m = Embedding(output_dim=vDim*mini_tDim,input_dim=tSize,input_length=maxTopic,weights=[rETM])(inputT)
    eTM_reshape = Reshape((1,maxTopic,vDim*mini_uDim))(embeddingT_m)
    eTM_pooling = MaxPooling2D(pool_size=(maxTopic,1),border_mode='valid')(eTM_reshape)
     
    eU_reshape = Reshape((uDim,))(eUV_pooling)
    eT_reshape = Reshape((tDim,))(eTV_pooling)

    embeddingC = Embedding(output_dim=vDim,input_dim=vSize,input_length=maxTotalLength,weights=[embeddings], trainable=False)(inputC)
     
    embeddingM = Embedding(output_dim=vDim,input_dim=vSize,input_length=maxComment*maxCmtLength,weights=[embeddings], trainable=False)(inputM)
    embeddingM_reshape = Reshape((maxComment,maxCmtLength,vDim))(embeddingM)
     
    ERV = Embedding(output_dim=uDim,input_dim=uSize,input_length=maxComment,weights=[rEUV])
    embeddingR_v = ERV(inputR)
     
    ERM = Embedding(output_dim=vDim*mini_uDim,input_dim=uSize,input_length=maxComment,weights=[rEUM])
    embeddingR_m = ERM(inputR)
     
    r1 = np.random.uniform(low=-rndBase,high=rndBase,size=(conSize,1,flength1,mini_uDim+mini_tDim))
    r2 = np.random.uniform(low=-rndBase,high=rndBase,size=(conSize,1,flength2,mini_uDim+mini_tDim))
    r3 = np.random.uniform(low=-rndBase,high=rndBase,size=(conSize,1,flength3,mini_uDim+mini_tDim))
    rb = np.random.uniform(low=-rndBase,high=rndBase,size=(conSize,))
     
    con1 = Convolution2D(conSize, flength1, mini_uDim+mini_tDim, weights=[r1,rb], activation='tanh',border_mode='valid')
    con2 = Convolution2D(conSize, flength2, mini_uDim+mini_tDim, weights=[r2,rb], activation='tanh',border_mode='valid')
    con3 = Convolution2D(conSize, flength3, mini_uDim+mini_tDim, weights=[r3,rb], activation='tanh',border_mode='valid')
     
    c1Avg = MaxPooling2D(pool_size=(maxTotalLength-flength1+1,1),border_mode='valid')
    c2Avg = MaxPooling2D(pool_size=(maxTotalLength-flength2+1,1),border_mode='valid')
    c3Avg = MaxPooling2D(pool_size=(maxTotalLength-flength3+1,1),border_mode='valid')
     
    c1AvgM = MaxPooling2D(pool_size=(maxCmtLength-flength1+1,1),border_mode='valid')
    c2AvgM = MaxPooling2D(pool_size=(maxCmtLength-flength2+1,1),border_mode='valid')
    c3AvgM = MaxPooling2D(pool_size=(maxCmtLength-flength3+1,1),border_mode='valid')
     
    cReshape = Reshape((conSize,))
     
    RU = Reshape((vDim,mini_uDim))
    RT = Reshape((vDim,mini_tDim))
    UTC_Reshape = Reshape((1,maxTotalLength,mini_uDim+mini_tDim))
    UTC_Reshape_cmt = Reshape((1,maxCmtLength,mini_uDim+mini_tDim))
    UTC_upper_Reshape = Reshape((1,conSize+uDim+tDim))
     
    t4 = time.time()
    print 'Initialization Elapse: ', t4-t3
     
    CMT=[]
    for i in range(maxComment):
        Mi = Lambda(getCmtWrapper(i),output_shape=getCmtWrapper_output_shape)(embeddingM_reshape)
        UMi = Lambda(getCmtrWrapper(i),output_shape=getCmtWrapper_output_shape)(embeddingR_m)
        UVi = Lambda(getCmtrWrapper(i),output_shape=getCmtWrapper_output_shape)(embeddingR_v)
          
        RUi = RU(UMi)
        UCi = merge([Mi,RUi],mode='dot',dot_axes=(2,1))
        RTi = RT(eTM_pooling)
        TCi = merge([Mi,RTi],mode='dot',dot_axes=(2,1))
        UTCi = merge([UCi,TCi],mode='concat')
        UTC_Ri = UTC_Reshape_cmt(UTCi)
          
        con1outi = con1(UTC_Ri)
        con2outi = con2(UTC_Ri)
        con3outi = con3(UTC_Ri)
          
        c1i = c1AvgM(con1outi)   #average of words
        c2i = c2AvgM(con2outi)
        c3i = c3AvgM(con3outi)
          
        c1_reshapei=cReshape(c1i)
        c2_reshapei=cReshape(c2i)
        c3_reshapei=cReshape(c3i)
              
        conavgi = merge([c1_reshapei,c2_reshapei,c3_reshapei],mode='ave')
        UTC_upperi = merge([conavgi,UVi,eT_reshape],'concat')
        UTC_upper_Ri = UTC_upper_Reshape(UTC_upperi)
        CMT.append(UTC_upper_Ri)
          
    CMT_concate = merge(CMT,mode='concat',concat_axis=1)
    CMT_concate_reshape = Reshape((1,maxComment,conSize+uDim+tDim))(CMT_concate)
    CMT_pooling = MaxPooling2D(pool_size=(maxComment,1),border_mode='valid')(CMT_concate_reshape)
    CMT_Reshape = Reshape((conSize+uDim+tDim,))(CMT_pooling)
     
    RUj = RU(eUM_pooling)
    UC = merge([embeddingC,RUj],mode='dot',dot_axes=(2,1))
    RTj = RT(eTM_pooling)
    TC = merge([embeddingC,RTj],mode='dot',dot_axes=(2,1))
    UTC = merge([UC,TC],mode='concat')
    UTC_Rj = UTC_Reshape(UTC)
 
    con1out = con1(UTC_Rj)
    con2out = con2(UTC_Rj)
    con3out = con3(UTC_Rj)
     
    c1 = c1Avg(con1out)   #average of words
    c2 = c2Avg(con2out)
    c3 = c3Avg(con3out)
     
    c1_reshape=cReshape(c1)
    c2_reshape=cReshape(c2)
    c3_reshape=cReshape(c3)
         
    conavg = merge([c1_reshape,c2_reshape,c3_reshape],mode='ave')
 
    t5 = time.time()
    print 'Sentences Processing Elapse: ', t5-t4
     
    UTMC_upper = merge([conavg,eU_reshape,eT_reshape,CMT_Reshape],'concat')
    rD  = np.random.uniform(low=-rndBase,high=rndBase,size=(conSize+uDim+tDim+conSize+uDim+tDim,lSize))
    rDb = np.random.uniform(low=-rndBase,high=rndBase,size=(lSize,))

    predict = Dense(lSize,activation='softmax',weights=[rD,rDb])(UTMC_upper)
     
    model = Model(input=[inputU,inputT,inputC,inputM,inputR],output=predict)
    ag = Adagrad(lr=tryAssign(dicPar,'lr'))
    model.compile(ag, 'categorical_crossentropy',['accuracy'])

    stop = EarlyStopping(monitor='val_acc',patience=tryAssign(dicPar,'patience'),mode='max')
    save = ModelCheckpoint(tryAssign(dicFile,'save_each'))
    model.fit([inputT1,inputT2,inputT3,inputT4,inputT5],to_categorical(llistTrain,lSize),callbacks=[stop,save],batch_size=tryAssign(dicPar,'batch_size'),nb_epoch=tryAssign(dicPar,'max_epoch'),validation_data=([inputT1Dev,inputT2Dev,inputT3Dev,inputT4Dev,inputT5Dev],to_categorical(llistDev,lSize)))

    t6 = time.time()
    print 'Fitting Elapse: ', t6-t5
    model.save_weights(tryAssign(dicFile,'save_final'))
       
    classes = model.predict([inputT1Test,inputT2Test,inputT3Test,inputT4Test,inputT5Test],batch_size=tryAssign(dicPar,'batch_size'))
    prediction = predictFromClass(classes)
    
    print precision_recall_fscore_support(llistTest,prediction,average=None,labels=[0,1,2])    

    trues = findTrues(llistTest, prediction)
     
    with open(tryAssign(dicFile,'save_pickle'), 'w') as f:
        pickle.dump(trues,f)