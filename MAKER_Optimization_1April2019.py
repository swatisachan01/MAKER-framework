import pandas as pd
import numpy as np
from functools import reduce
from operator import mul
from itertools import product
import time
start = time.time()

np.seterr(divide='ignore', invalid='ignore') #ignore division by invalide and 0 in 'interrelation = new_df[-1].values/ss1.values'
Data_file = r'C:/MyResearch Folder/WORK FOLDER/Asthama data/All the Dataset (Asthma) Cleaned by Huaying Zhu 2015.xlsx'
xls_file = pd.ExcelFile(Data_file)
print(xls_file.sheet_names)
data1 = xls_file.parse('maker_data') ##sheet to import
header = list(data1.columns.values)
print("List of attributes:" + str(header))

data1.iloc[:,-1] = data1.iloc[:,-1].fillna('UNKNOWN') #fill NAN in last attribute with UNKNOWN

idx = data1.iloc[:,-1].unique() #Put UNKNOWN always in first in the list
if 'UNKNOWN' in idx:
    idx.partition(np.where(idx=='UNKNOWN')[0][0])  
#print(idx)

######################### PREPARE DATA #############################
#COMPLETE DATA
m = data1.iloc[:, 0:len(header)-1].isnull().any(1)
complete_data = data1[~m]
complete_data = complete_data.astype(str)  #make dataframe string to avoid error in contigency table
#print(complete_data)

#INCOMPLETE DATA
data1 = data1.astype(str).replace('nan',np.nan) #convert datafarme to string exclusing nan values. String data ensures correct frequency/contigency table
#print(data1)

####### Initilaize Weight, Reliability and Joint Support Parameters #############################
# List of attribute single and combined to find initialized parameter
variable = [] #List store all the variables, single and joint: example d =[['variable1'],['variable2'],['variable1','variable2']]
for i in range(0,len(header)-1): #This loop put single variable in list d # len(header)-1 excludes the last consequence variable
    temp = []
    temp.append(data1.iloc[:,i])
    variable.append(temp)    
for j in range(0,len(header)-2): #This loop put joint variable in a list d
    d1=[]
    for i in range(0,j+2):    
        d1.append(data1.iloc[:,i])
    variable.append(d1)


#Intialize weight of evidence pointing to hypothesis h. Initial weight is equal to 1     
initial_weight = []
for i in range(0,len(variable)):
    c1= pd.crosstab([complete_data.iloc[:,-1]],variable[i],dropna=False) #1)Frequency table #last column is output variable
    print(c1)
    c1 = c1.reindex(idx, fill_value=0)
    c1 = c1.div(c1.sum(axis=1), axis=0) #2)Likelihood table
    prob1 = c1.div(c1.sum(axis=0), axis=1) #3)Probability table    
    prob1[:] = 1
    initial_weight.append(prob1)
'''
#Initialize reliability
initial_reliability = initial_weight.copy()
#print(initial_reliability)

#Intialize joint degree of support pointing to hypothesis h. Initial joint degree of support is equal to 1  
l = len(header)-1
initial_degree_joint_support = initial_weight[l:2*l-1].copy()
#print(initial_degree_joint_support) 


############# MAKER ####################
def Interrelation():  #Interrelation is calculated from subset (complete data)
    #single and joint probability   
    global Temp_Prob_df
    Temp_Prob_df = [] #stores both single and joint probability from list d
    for i in range(0,len(variable)):
        
        df2=pd.DataFrame(variable[i]).T #make list to dataframe
        df3=pd.concat([df2, data1.iloc[:,-1]],axis=1) #combine columns of two dataframes
        m = df3.iloc[:, 0:len(df3.columns)-1].isnull().any(1) #find missing input
        sub_set = df3[~m] #inxonplete data
        sub_set = sub_set.astype(str) #make every thing as string
       
        
        c1 = pd.crosstab([sub_set.iloc[:,-1]],variable[i],dropna=False) #1)Frequency table #last column is output variable
        c1 = c1.reindex(idx, fill_value=0) #reindex, take index of any of the dataframes of incomplete dataset
        c1 = c1.div(c1.sum(axis=1), axis=0) #2)Likelihood table
        prob1 = c1.div(c1.sum(axis=0), axis=1) #3)Probability table
        prob1 = prob1.fillna(0) #replace nan with 0, some of the combinations does not exist in the data
        Temp_Prob_df.append(prob1) #contain both single and joint probability
    #print(Temp_Prob_df)
    
    #2)Interrelation from subset complete data    
    #Denominator
    Multi_deno = [] # Empty list of denominator (multiplication) in interrelation
    start = len(header)-1  #excluding last element 
    end = len(Temp_Prob_df)
    
    for i in range(start,end):
        s1 = len(header) - 2
        
        if i == start:
            d1,d2 = Temp_Prob_df[0],Temp_Prob_df[1]
        else:  
            nn = Temp_Prob_df[i-1].copy() #copy joint probability p[i-1] to nn dataframe, so that multilevel structure of prob_df does not change                                       
            nn.columns = nn.columns.map('|'.join) #merge columns of joint probability
            d1,d2 = nn,Temp_Prob_df[i-s1] #multiple joint with single. First joint p[i-1]*single[i-s1] 
             
        deno_inter=pd.concat({k: reduce(mul, (d[c] for d, c in zip([d1, d2], k)))for k in product(d1, d2)}, axis=1)   
        Multi_deno.append(deno_inter)     
    #print(Multi_deno)
          
    #Division by numpy array to find the interrelation
    global Interrelation 
    Interrelation = []  #empty list of all interrelation datafarme
    for i in range(0,len(Multi_deno)):
        start1 = len(header)-1    
        np.seterr(divide='ignore', invalid='ignore') #ignore division by zero
        temp_inter = Temp_Prob_df[start1+i].values/Multi_deno[i].values #division
        temp_inter = np.nan_to_num(temp_inter) #replace nan with 0        
        temp_inter = pd.DataFrame(temp_inter,columns=Temp_Prob_df[start1+i].columns, index=Temp_Prob_df[start1+i].index) #add multilevel and index in dataframe
        Interrelation.append(temp_inter)
   


def MAKER_func1():
    ########## COMPLETE DATA ##################
    #1)Complete data: Single and joint probability    
    global Prob_df
    Prob_df = [] #stores both single and joint probability from list d
    for i in range(0,len(variable)):
        c1 = pd.crosstab([complete_data.iloc[:,-1]],variable[i],dropna=False) #1)Frequency table #last column is output variable
        c1 = c1.reindex(idx, fill_value=0) #reindex, take index of any of the dataframes of incomplete dataset
        c1 = c1.div(c1.sum(axis=1), axis=0) #2)Likelihood table
        prob1 = c1.div(c1.sum(axis=0), axis=1) #3)Probability table
        prob1 = prob1.fillna(0) #replace nan with 0, some of the combinations does not exist in the data
        Prob_df.append(prob1) #contain both single and joint probability
    #print(Prob_df)
        
    ##########INCOMPLETE DATA##################
    #1) Probability Mass from incomplete data likelihood and basic probability
    global single_var_incomplete
    single_var_incomplete = [] #List store all single variables: example single_var_incomplete =[['variable1'],['variable2']
    for i in range(0,len(header)-1): #This loop put single variable in list d # len(header)-1 excludes the last consequence variable
        temp = []
        single_INc = data1.iloc[:,i]
        temp.append(single_INc)
        single_var_incomplete.append(temp)
    #print(single_var_incomplete)
    #print(len(single_var_incomplete))   
    
    global Prob_df_INcomplete
    Prob_df_INcomplete = [] #stores both single and joint probability from list d
    for i in range(0,len(single_var_incomplete)):
        c2 = pd.crosstab([data1.iloc[:,-1]],single_var_incomplete[i],dropna=True) #1)Frequency table #last column is output variable    
        c2 = c2.reindex(idx, fill_value=0)
        c2 = c2.div(c2.sum(axis=1), axis=0) #2)Likelihood table
        prob2 = c2.div(c2.sum(axis=0), axis=1) #3)Probiability table
        prob2 = prob2.fillna(0) #replace nan with 0, some of the combinations does not exist in the data
        Prob_df_INcomplete.append(prob2) #contain both single and joint probability
    #print(Prob_df_INcomplete)
    


def MAKER_func2(e_weight,e_reliability,e_joint_support):    
    #First find probability mass for single variable and they are combined by MAKER later 
    Prob_Mass = [] #empty list of probability mass     
    for i in range(0,len(single_var_incomplete)):    
        mass_i =  e_weight[i]*Prob_df_INcomplete[i]
        mass_i = mass_i.fillna(0) #replace nan with 0
        Prob_Mass.append(mass_i)        
    #print(Prob_Mass) 
    
    ############# Reliability of Evidence ###################
    #Reliability of Evidence = 1 - Summation(reliability_evidence_pointing_to_state * complete probability 
    Residual_Relability_Evidence = []
    for i in range(0,len(initial_reliability)):
        temp_r = e_reliability[i]*Prob_df[i]
        temp_r = 1.00 - (temp_r.sum(axis=0).to_frame().T)
        temp_r.values[np.isclose(temp_r.values, 0)] = 0  #avoid very samll positive number. Example, 1.110223e-16
        Residual_Relability_Evidence.append(temp_r) 
     
        
    #########Joint Probability Mass ##############
    
    #4) Joint Probability Mass
    from operator import add
    j = 0  # start point for interrelation and joint degree of support
    start = len(header)-1  #excluding last element 
    end = len(Prob_df)
    
    for i in range(start,end):
         
        #print('')
        #print('i is equal to: '+str(i))    
        
        s2 = len(header) - 2
        if i == start:
            d1,d2 = Prob_Mass[0],Prob_Mass[1]
    
    
            aa1 = d1.mul(Residual_Relability_Evidence[0].loc[0],axis=1)
            aa2 = d2.mul(Residual_Relability_Evidence[1].loc[0],axis=1)
            part1 = pd.concat({k: reduce(add, (d[c] for d, c in zip([aa1, aa2], k)))for k in product(aa1, aa2)}, axis=1)  
            #print('part1')
            #print(part1)
        else:
            nn = Prob_Mass[i-1].copy() #copy joint probability p[i-1] to nn dataframe, so that multilevel structure of prob_df does not change                                       
            #nn.columns = nn.columns.map('|'.join) #merge columns of joint probability
            d1,d2 = nn,Prob_Mass[i-s2] #multiple joint with single. First joint p[i-1]*single[i-s1] 
            #print(d1)
            #print(d2)
            
            aa1 = d1.mul(Residual_Relability_Evidence[i-1].loc[0],axis=1)
            aa2 = d2.mul(Residual_Relability_Evidence[i-s2].loc[0],axis=1)
            #print(aa1)
            #print(aa2)
            part1 = pd.concat({k: reduce(add, (d[c] for d, c in zip([aa1, aa2], k)))for k in product(aa1, aa2)}, axis=1)  
            part1 = part1.values
            part1 = pd.DataFrame(part1,columns=Interrelation[j].columns, index=Prob_df[0].index) #add multilevel and index in dataframe                
            #print('part1')
            #print(part1)        
    
            
            
        ######### PART2############
        #first multiplication
        first=pd.concat({k: reduce(mul, (d[c] for d, c in zip([d1, d2], k)))for k in product(d1, d2)}, axis=1)   
        #print('first')
        #print(first)    
        
        if 'UNKNOWN' in idx:
            #second multiplication
            m1 = d1.copy()   #unknown values
            s = m1.iloc[0].values
            for i in range(0,len(idx)):
                m1.iloc[i] = s
            m1.iloc[0] = 0
            #print('m1')            
            #print(m1)
            
            
            m2 = d2.copy()  #unknown values
            s = m2.iloc[0].values
            for i in range(0,len(idx)):
                m2.iloc[i] = s
            m2.iloc[0] = 0
            #print('m2')    
            #print(m2)    
        
            second1 = pd.concat({k: reduce(mul, (d[c] for d, c in zip([d1, m2], k)))for k in product(d1, d2)}, axis=1)   
            second2 = pd.concat({k: reduce(mul, (d[c] for d, c in zip([m1, d2], k)))for k in product(d1, d2)}, axis=1)   
            #print('second1')
            #print(second1)
            #print('second2')
            #print(second2)
            
            #print('Interrelation')
            #print(Interrelation[j])
            
            #add dataframes together
            summation = sum([first, second1, second2])
            summation = summation.values    
            summation = pd.DataFrame(summation,columns=Interrelation[j].columns, index=Interrelation[0].index) #add multilevel and index in dataframe
            #print('summation')
            #print(summation)
        
        else:
            summation = first.copy()
            summation = summation.values
            summation = pd.DataFrame(summation,columns=Interrelation[j].columns, index=Interrelation[0].index)
            
            
        part2 =  Interrelation[j]*e_joint_support[j]*summation
        #print('part2')
        #print(part2)
        
        P_Mass = sum([part1, part2])
        #print('Probability Mass')
        #print(P_Mass)
        
        Prob_Mass.append(P_Mass)
       
        j = j+1
        
    #print(Prob_Mass)
    global Estimated_prob_mass
    Estimated_prob_mass = Prob_Mass[-1]
    Estimated_prob_mass = Estimated_prob_mass /  Estimated_prob_mass.sum() #Normalize 
    Estimated_prob_mass = Estimated_prob_mass.fillna(0) #replace nan with 0
    #print(Estimated_prob_mass)
    return Estimated_prob_mass


#Objective Function
#1) one hot encode complete data
# Get one hot encoding of last coloumn - consequence 
one_hot = pd.get_dummies(complete_data.iloc[:,-1]) 
droped_df = complete_data.drop(complete_data.columns[-1], axis=1)
Observed = droped_df.join(one_hot)
Observed = Observed.drop(Observed.columns[0:len(header)-1], axis=1) #remove excessive columns
Observed = Observed.reset_index(drop=True) #reset index and drop true index 
#print(Observed)


def objective_func():
    #2) Fill complete data by estimated Probability Mass
    Dict_Estimated_prob_mass = Estimated_prob_mass.T #transpose the generated probability 
    Dict_Estimated_prob_mass = Dict_Estimated_prob_mass.to_dict() #tranfer to dictionary format
    fill_generated_prob = pd.DataFrame(Dict_Estimated_prob_mass) #retransfer to dataframe format
    #print(fill_generated_prob)
    
    select_df = complete_data.drop(complete_data.columns[-1], axis=1) #select the columns which are used for joint probability 
    Estimated = fill_generated_prob.reindex(select_df.set_index(select_df.columns.tolist()).index).reset_index() #map data from fill_generated_prob to complete data
    Estimated = Estimated.drop(Estimated.columns[0:len(header)-1], axis=1) #remove excessive columns
    #print(Estimated)
    
    global objective_value
    objective_value = (((Observed - Estimated)**2).sum(axis=1).sum(axis = 0))/(2*len(Estimated)) ##(Observed - Estimated)**2 is substract two dataframe then square it. 
    print(objective_value)

Interrelation() 
MAKER_func1()
MAKER_func2(initial_weight,initial_reliability,initial_degree_joint_support)
#print(Interrelation[1])

####################################MAKER Optimization############################
from scipy.optimize import minimize

#Number of parameters to train
num_we_re = 0
for i in range(0,len(initial_weight)):
    num_we_re = num_we_re + initial_weight[i].size

num_support = 0
for i in range(0,len(initial_degree_joint_support)): 
    num_support = num_support + initial_degree_joint_support[i].size


#Initial parameter
initial_parameter = []
for i in range(0,len(initial_weight)):
    y = initial_weight[i].values
    f = y.flatten()
    initial_parameter.append(f)
    
for i in range(0,len(initial_reliability)):
    y = initial_reliability[i].values
    f = y.flatten()
    initial_parameter.append(f)

for i in range(0,len(initial_degree_joint_support)):
    y = initial_degree_joint_support[i].values
    f = y.flatten()
    initial_parameter.append(f)  
initial_parameter = np.concatenate(initial_parameter)
#print(initial_parameter)

num = []
a = 0
for i in range(0,len(initial_weight)):
    a = a + initial_weight[i].size
    num.append(a)
for i in range(0,len(initial_reliability)):
    a = a + initial_reliability[i].size
    num.append(a)
for i in range(0,len(initial_degree_joint_support)):
    a = a + initial_degree_joint_support[i].size
    num.append(a)
#print(num)

MAKER_func1() # run this function first. Not required in optimization
def optimization(x):
    x = np.split(x, num)
    #print(parameter)
    
    e_weight = []
    for i in range(0,len(initial_weight)):
        parameter1 = x[i].reshape(initial_weight[i].shape)
        parameter1 = pd.DataFrame(parameter1,columns=initial_weight[i].columns, index=initial_weight[i].index) #add multilevel and index in dataframe                
        e_weight.append(parameter1)
    #print(weight)
    
    e_reliability = []
    j = 0    
    for i in range(len(initial_weight),len(initial_weight)*2):
        parameter2 = x[i].reshape(initial_reliability[j].shape)
        parameter2 = pd.DataFrame(parameter2,columns=initial_reliability[j].columns, index=initial_reliability[j].index) #add multilevel and index in dataframe                
        e_reliability.append(parameter2)
        j = j+1
    #print(reliability)  
    
    e_joint_support = []    
    j = 0    
    for i in range(len(initial_weight)*2,(len(initial_weight)*2)+len(initial_degree_joint_support)):
        parameter3 = x[i].reshape(initial_degree_joint_support[j].shape)
        parameter3 = pd.DataFrame(parameter3,columns=initial_degree_joint_support[j].columns, index=initial_degree_joint_support[j].index) #add multilevel and index in dataframe                
        e_joint_support.append(parameter3)
        j = j+1
    #print(joint_support) 
    MAKER_func2(e_weight,e_reliability,e_joint_support)
    objective_func()
    return objective_value


#####bound constrain#######
b1 = [0.0, 1.0]
constrain_bound = []
for i in range(0,(num_we_re*2 + num_support)):
    constrain_bound.append(b1)  


solution = minimize(optimization,initial_parameter,method='L-BFGS-B',bounds=constrain_bound,options={"disp": True,'eps':3e-1,'maxiter':6}) #"disp": True
x = solution.x
print('Final Objective function value: ' + str(optimization(x)))   
print(x)
print('')
print('Number of weight parameters: ' +str(num_we_re))
print('Number of reliability parameters: ' +str(num_we_re))
print('Number of joint support parameters: ' +str(num_support))
print('Total number of parameters: '+str(num_we_re*2 + num_support))
print('')

##restruct numpy array of trained parameter
def trained_parameter(x):
    x = np.split(x, num)
    t_weight = []
    for i in range(0,len(initial_weight)):
        parameter1 = x[i].reshape(initial_weight[i].shape)
        parameter1 = pd.DataFrame(parameter1,columns=initial_weight[i].columns, index=initial_weight[i].index) #add multilevel and index in dataframe                
        t_weight.append(parameter1)
    print(t_weight)
    
    t_reliability = []
    j = 0    
    for i in range(len(initial_weight),len(initial_weight)*2):
        parameter2 = x[i].reshape(initial_reliability[j].shape)
        parameter2 = pd.DataFrame(parameter2,columns=initial_reliability[j].columns, index=initial_reliability[j].index) #add multilevel and index in dataframe                
        t_reliability.append(parameter2)
        j = j+1
    print(t_reliability)  
    
    t_joint_support = []    
    j = 0    
    for i in range(len(initial_weight)*2,(len(initial_weight)*2)+len(initial_degree_joint_support)):
        parameter3 = x[i].reshape(initial_degree_joint_support[j].shape)
        parameter3 = pd.DataFrame(parameter3,columns=initial_degree_joint_support[j].columns, index=initial_degree_joint_support[j].index) #add multilevel and index in dataframe                
        t_joint_support.append(parameter3)
        j = j+1
    print(t_joint_support) 
trained_parameter(x)    

print('After Training: Combined Probability')
print(Estimated_prob_mass)

end = time.time()
print('Total execution time is (in minutes):')
print((end - start)/60)

##export transformed data
Dict_Estimated_prob_mass = Estimated_prob_mass.T #transpose the generated probability 
Dict_Estimated_prob_mass = Dict_Estimated_prob_mass.to_dict() #tranfer to dictionary format
fill_generated_prob = pd.DataFrame(Dict_Estimated_prob_mass) #retransfer to dataframe format
select_df = complete_data.drop(complete_data.columns[-1], axis=1) #select the columns which are used for joint probability 
Estimated = fill_generated_prob.reindex(select_df.set_index(select_df.columns.tolist()).index).reset_index() #map data from fill_generated_prob to complete data
print(Estimated)   
'''