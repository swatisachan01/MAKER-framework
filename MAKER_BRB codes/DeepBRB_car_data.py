# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 10:57:20 2019

@author: a47118ss
"""

import numpy as np
import pandas as pd
from itertools import accumulate

MK_input = [0.9244288224956063, 0.8813708260105448, 0.867311072056239, 0.7847100175746924]
#MK_input =[0.7777777777777778, 0.7777777777777778, 0.7777777777777778]
N = 2 #always N equal to or more than 2
stop_level = 1

Data_file= r'\\nask.man.ac.uk/home$/Desktop/Interactive BRB with MAKER/Dataset/transfomed_mk/tf_cancer.xlsx'
#Data_file= r'\\nask.man.ac.uk/home$/Desktop/Interactive BRB with MAKER/Dataset/transfomed_mk/tf_car.xlsx'

mk_df = pd.read_excel(Data_file)
#print(mk_df)

n_c = len(mk_df[mk_df.columns.values[-1]].unique()) #output is last column, get number of unique values or num of consequence RV
RV_output = mk_df[mk_df.columns.values[-1]].unique() #all unique values of last column is referential value
df_tf = mk_df.iloc[:,0:-1].values #dataframe to numpy array 
L3 = np.array([mk_df.iloc[:,(i*n_c):((i+1)*n_c)].values for i in range(len(MK_input))])
#print(L3)

    

def Level_and_Initial_Parameters(N,MK_input,stop_level,RV_output):
    #1)Levels in BRB
    level = []
    while len(MK_input) > stop_level: #stop when number of elements in level are less than stop_level
        if (N % 2) == 0 and (len(MK_input)%2) == 0: #when both N and length of level (numbers of elements) are even
            subList = [MK_input[n:n+N] for n in range(0, len(MK_input), N)]     
        else:
            N = N+1 #else cut level in odd numbers
            subList = [MK_input[n:n+N] for n in range(0, len(MK_input), N)]
        MK_input = []
        for i in range(len(subList)):
            MK_input.append(sum(subList[i])/len(subList[i]))
        level.append(subList)
        #print(subList)
    #print(level)

    #2)Normalize the initial accuracy of each level to fill the initial belief degree
    global Normalized_acc_level
    Normalized_acc_level = []
    for i in range(len(level)):
        temp_level = []
        for j in range(len(level[i])):
            norm_temp = []
            norm_temp =  [x / sum(level[i][j]) for x in level[i][j]] #normalization
            temp_level.append(norm_temp)
        Normalized_acc_level.append(temp_level)
    #print(Normalized_acc_level)   

    #3) Rules: Set of rules in each element of BRB
    global Rule_All_Level
    Rule_All_Level = []
    for i in range(len(level)):
        temp2 = []
        for j in range(len(level[i])):
            RV_temp=[]
            RV_temp += len(level[i][j]) * [RV_output]
            r_l = np.array(np.meshgrid(*RV_temp), dtype=object).T.reshape(-1,len(RV_temp)) #rules combination
            temp2.append(r_l)
        Rule_All_Level.append(temp2)
    #print(Rule_All_Level)   

    #4) Intial: Attribute weight and rule weight
    i_att_weight = []
    i_rl_weight = []
    for i in range(len(Rule_All_Level)):
        temp_att_w = [] #empty for temporary attribute weight
        temp_rule_w = [np.ones(len(Rule_All_Level[i][j])) for j in range(len(Rule_All_Level[i]))] #temporary rule weight
        for j in range(len(Rule_All_Level[i])):            
            att_wgt_temp =  np.copy(Rule_All_Level[i][j])  #attribute weight
            att_wgt_temp.fill(1.0)
            temp_att_w.append(att_wgt_temp) #attribute weight
        i_rl_weight.append(np.array(temp_rule_w))
        i_att_weight.append(np.array(temp_att_w))
    #print(i_rl_weight)
    #print(i_att_weight)

    #5)Initial: Belief Degree
    i_belief_consequence = [] #belief-degree list
    for i in range(len(Normalized_acc_level)): #level
        belief_temp = []  #temporary belief-degree list
        for j in range(len(Normalized_acc_level[i])): #set in a level
            r_l = np.array(Rule_All_Level[i][j]) #rule in each level
            acc_l = np.array(Normalized_acc_level[i][j])
            s = pd.DataFrame(r_l).stack().reset_index(name='val')
            s['level_1'] = acc_l[s['level_1']]
            ff = s.pivot_table(index='level_0', columns='val', values='level_1', aggfunc='sum', fill_value=0)
            belief_temp.append(ff.values.round(decimals=4))
        i_belief_consequence.append(np.array(belief_temp))
    #print(i_belief_consequence) 
    
    #6) referetial values
    global rv_level    
    rv_level = []
    for l in range(len(Normalized_acc_level)):
        s = [np.repeat([RV_output],repeats=len(Normalized_acc_level[l][g]), axis=0) for g in range(len(Normalized_acc_level[l]))]
        rv_level.append(s)
    #print(rv_level) 
    
    #7) level0 DATA
    #Cut L3 to make group for level 0 as input data, l0_L3
    global l0_L3
    cut_lv0 = list(accumulate([len(Normalized_acc_level[0][i]) for i in range(len(Normalized_acc_level[0]))][0:-1]))
    l0_L3 = np.split(L3, cut_lv0)
    #print(l0_L3)
    #print(len(l0_L3[0]))

    #8) observed values of belief
    global obs
    obs = pd.get_dummies(mk_df.iloc[:,-1]) #real belief/probability column
    obs = obs[RV_output] #match the column order as RV list
    #print(obs)       
    
    return level,Rule_All_Level,i_att_weight,i_rl_weight,i_belief_consequence,l0_L3,obs  

gamma = Level_and_Initial_Parameters(N,MK_input,stop_level,RV_output)  
#print(gamma)


def Inference_Engine(datai,r,RV,att_weight,rl_weight,belief_consequence):  
    print(datai.shape)
    print(datai)
    arr_data = np.hstack((datai)).reshape(datai.shape[1],datai.shape[0],datai.shape[2]) #reshape data in row. input data is shapes in columns to divide the level. However here it must be in rows 

    norm_att_weight = att_weight/att_weight.max(axis=1)[:,None]    
    match = np.array([np.array(np.meshgrid(*arr_data[m]), dtype=object).T.reshape(-1,len(RV)) for m in range(len(arr_data))])
    #power_match_att = np.array([np.power(match[m], norm_att_weight) for m in range(len(arr_data))])
    power_match_att = np.power(match, norm_att_weight)
    product1 = np.array([np.prod(power_match_att[m], axis=1) for m in range(len(arr_data))])  
    #non_normalize_activation = np.array([np.multiply(product1[m],rl_weight) for m in range(len(arr_data))])
    non_normalize_activation = np.multiply(product1,rl_weight)
    #print(non_normalize_activation)
    #act_weight = np.array([np.true_divide(non_normalize_activation[m], np.sum(non_normalize_activation[m])) for m in range(len(arr_data))])    
    act_weight = np.array([np.true_divide(non_normalize_activation[m], np.sum(non_normalize_activation[m])) for m in range(len(arr_data))])
    
    num_part2 = np.prod(1 - act_weight, axis=1)
    num_part1 = []
    for i in range(0,len(act_weight)):
        temp = []
        for n in range(0,len(belief_consequence[0])): 
            prod2 = 1
            for k in range(0,len(r)): 
                prod2 = prod2 * ((act_weight[i][k]*belief_consequence[k][n])+1-act_weight[i][k])
            temp.append(prod2)
        num_part1.append(temp)
    num_part1 = np.asarray(num_part1)       
    deno_part1 = np.sum(num_part1, axis=1)
    deno_part2 = (len(belief_consequence[0])-1) * num_part2
    deno_part3 = np.copy(num_part2)
    deno_final = deno_part1 - deno_part2 - deno_part3
    num_final = np.subtract(num_part1, num_part2.reshape((-1, 1)))
    aggregate_belief = np.true_divide(num_final,deno_final.reshape((-1, 1)))
    #print('belief output: '+str(aggregate_belief))
    return aggregate_belief

def H_BRB(x,y,z):
    #x: att_weight, y :rule weight, z: belief consequence
    for l in range(len(Normalized_acc_level)):
        if l == 0: #first level with MAKER data
            #print(l0_L3[0])
            out_lv0 = [Inference_Engine(l0_L3[0],Rule_All_Level[0][g],rv_level[0][g],x[0][g],y[0][g],z[0][g]) for g in range(len(Normalized_acc_level[0]))]
        if l > 0: #levels higher than 0 
            out_lv = []
            if l == 1: 
                l_h = np.array([[next(iter(out_lv0)) for _ in sublist] for sublist in Normalized_acc_level[1]])   #cut outside beacuse you donot want to repeat
    
            out_lv = [Inference_Engine(l_h[g],Rule_All_Level[l][g],rv_level[l][g],x[l][g],y[l][g],z[l][g]) for g in range(len(Normalized_acc_level[l]))]
            l_h = np.array([[next(iter(out_lv)) for _ in sublist] for sublist in Normalized_acc_level[l]])
        if len(Normalized_acc_level) == 1: #if only one level
            out_lv = out_lv0
    est = pd.DataFrame(out_lv[0],columns=RV_output) #the last level will have only one group, take the 0th indexed output from last aggregated beliefs
    error = (((est-obs)**2).sum(axis=1).sum(axis = 0))/(2*len(est))
    print(error) #retun objective function
    return error, est
H_BRB(gamma[2],gamma[3],gamma[4])

def sample(s1,s2,s3):
    #s1: rule weight, s2:att_weight,s3:belief_consequence
    #####Initial parameters
    #1) flatten rule weight
    rl1 = np.array([s1[i].flatten() for i in range(len(s1))])
    rl_flat = np.concatenate(rl1).ravel()
    #print(rl_flat)
    
    #2) flatten att weight
    att1 = np.array([s2[i].flatten() for i in range(len(s2))])
    att_flat = np.concatenate(att1).ravel()
    #print(att_flat)
    
    #3) flatten belief 
    bl1 = np.array([s3[i].flatten() for i in range(len(s3))]) 
    bl_flat = np.concatenate(bl1).ravel()
    #print(bl_flat)
    
    #4) flat sample for optimization as initial starting point
    flat_initial_value = np.concatenate((rl_flat,att_flat,bl_flat), axis=None)
    #print(flat_initial_value)  
    return rl_flat,att_flat,bl_flat,flat_initial_value
num_con = sample(gamma[3],gamma[2],gamma[4])


def cut_index(s1,s2,s3):    
    #s1: rule weight, s2:att_weight,s3:belief_consequence
    #Cutting
    #1) cut rule weight
    
    r_cut_g = list(accumulate(sum([[len(s1[i][j]) for j in range(len(s1[i]))] for i in range(len(s1))], []))) #cut up to second list array that is 0:-1
    r_cut_lv = list(accumulate([len(s1[i]) for i in range(len(s1))][0:-1]))
    
    #2) cut attribute weight
    a_cut_g = list(accumulate(np.array(sum([[np.product(s2[i][j].shape) for j in range(len(s2[i]))] for i in range(len(s2))], []))))
    a_cut_lv = list(accumulate([len(s2[i]) for i in range(len(s2))][0:-1]))

    #3) cut belief consequence
    bl_cut_g = list(accumulate(np.array(sum([[np.product(s3[i][j].shape) for j in range(len(s3[i]))] for i in range(len(s3))], []))))
    bl_cut_lv = list(accumulate([len(s3[i]) for i in range(len(s3))][0:-1]))
    
    
    #4) cut belief constrains in belief_contrains() function 
    cut_bl_con = sum([[len(s1[i][j]) for j in range(len(s1[i]))] for i in range(len(s1))],[])
    
    #global para_cut 
    group_para_cut =  [r_cut_g[-1],r_cut_g[-1]+a_cut_g[-1],r_cut_g[-1]+a_cut_g[-1]+bl_cut_g[-1]] #parameter cut    
    
    return r_cut_g, r_cut_lv, a_cut_g, a_cut_lv, bl_cut_g, bl_cut_lv, cut_bl_con, group_para_cut          
ind = cut_index(gamma[3],gamma[2],gamma[4]) #initial values


def bound_constrains():
    bound_con = np.repeat(np.array([[0.001, 1]]),len(num_con[-1]),axis=0)  #num_con[-1] is flat_initial_value
    return bound_con
all_bounds = bound_constrains()



def eq_belief_constrains(z):
    def g(v):
        test = np.split(v, ind[-1]) # split flat sample           
        bl_gp = np.split(test[2], ind[4][0:-1])                    
        bl_c = np.concatenate(np.array([bl_gp[i].reshape(ind[6][i],-1) for i in range(len(ind[6]))]))  #split the beliefs in groups then send each array as sum = 1
        return np.sum(bl_c[z])-1
    return g
con = [{'type': 'eq', 'fun': eq_belief_constrains(z)} for z in range(ind[0][-1])]   #ind[4][-1] is number of beliefs 

def optimize_H_BRB(v):  #v is the flat sample of a parameters in optimization 
    v = np.asarray(v)
    
    #test[0]:rule weight, test[1]:attribute weight, test[2]:belief consequence
    test = np.split(v, ind[-1])  
    #print(test)
    
    #rule is test[0]
    rl_gp = np.split(test[0], ind[0])
    rl_w = np.split(rl_gp[0:-1], ind[1]) #rl_gp[0:-1] avoid last spilt
    #print(rl_w) 
    
    #attribute is test[1]
    att_gp = np.split(test[1], ind[2][0:-1])
    att_lv = np.split(att_gp, ind[3])
    att_w = np.array([np.array([att_lv[i][j].reshape(len(rl_w[i][j]),-1) for j in range(len(rl_w[i]))]) for i in range(len(rl_w))])

    #belief is test[2]
    bl_gp = np.split(test[2], ind[4][0:-1])
    bl_lv = np.split(bl_gp, ind[5])
    bl_c = np.array([np.array([bl_lv[i][j].reshape(len(rl_w[i][j]),-1) for j in range(len(rl_w[i]))]) for i in range(len(rl_w))])
    #print('belief after cutting', bl_c)
    
    objective_value = H_BRB(att_w,rl_w,bl_c)
    print(objective_value[0])
    return objective_value[0]

#optimize_H_BRB(num_con[-1])

def call_optimizer():
    from scipy.optimize import minimize
    solution = minimize(optimize_H_BRB,num_con[-1],method='SLSQP',bounds=all_bounds, constraints=con,options={'ftol': 1e-06})
    train_parameter = solution.x
    #print(train_parameter)
    return train_parameter
opt_result = call_optimizer()

def print_dataparameters(v):
    #test[0]:rule weight, test[1]:attribute weight, test[2]:belief consequence
    test = np.split(v, ind[-1])     
    #rule is test[0]
    rl_gp = np.split(test[0], ind[0])
    rl_w = np.split(rl_gp[0:-1], ind[1]) #rl_gp[0:-1] avoid last spilt
    #attribute is test[1]
    att_gp = np.split(test[1], ind[2][0:-1])
    att_lv = np.split(att_gp, ind[3])
    att_w = np.array([np.array([att_lv[i][j].reshape(len(rl_w[i][j]),-1) for j in range(len(rl_w[i]))]) for i in range(len(rl_w))])

    #belief is test[2]
    bl_gp = np.split(test[2], ind[4][0:-1])
    bl_lv = np.split(bl_gp, ind[5])
    bl_c = np.array([np.array([bl_lv[i][j].reshape(len(rl_w[i][j]),-1) for j in range(len(rl_w[i]))]) for i in range(len(rl_w))])
    #print('belief after cutting', bl_c)
    
    df_est = H_BRB(att_w,rl_w,bl_c)
    print(df_est[1])
    #export final beliefs or ckassified classes to xlsx sheet
    from pandas import ExcelWriter
    writer = ExcelWriter('C:/MyResearch Folder/WORK FOLDER/MAKER DATA1.xlsx')
    df_est[1].to_excel(writer,'s1')
    writer.save()
   
    print(rl_w, att_w, bl_c) #df_est[1] is est dataframe of beliefs
print_dataparameters(opt_result)
