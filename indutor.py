# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 00:01:59 2021

@author: valteresj
"""
import pandas as pd
import numpy as np
from info_gain import info_gain
import re
from tqdm import tqdm
from optbinning import OptimalBinning
import scipy as stat
path='C:/Users/valteresj/Documents/Projetos/automl'


dt=pd.read_csv(path+'/bank-full.csv',sep=';')

# path='C:/Users/valteresj/Documents/Projetos/saude'
# dt=pd.read_excel(path+'/dados_v1.xlsx')

target='y'
n_bins=5
label='yes'
level_rule=4
perc=0.01
t_lift=1.2
path='C:/Users/valteresj/Documents/Projetos/Indutor_regras'

def interval_woe(dt,target,label,n_bins=5,path=None):
    
    
    ref_var=dt[dt.columns[dt.columns!=target]].dtypes
    y = np.where(dt[target]==label,1,0)
    feat={}
    for i in ref_var[ref_var!='object'].index:
        optb = OptimalBinning(name=i, dtype="numerical", solver="cp",min_n_bins=3,max_n_bins=n_bins)
        optb.fit(dt[i].values, y)
        cut=list(np.unique(optb.splits))
        cut.insert(0,dt[i].min())
        cut.append(dt[i].max())
        if len(cut)>2:
            feat[i]=cut
            vv=pd.cut(dt[i],cut, include_lowest=True).astype(str)
            ref_max=[j for j in vv.unique() if j.find(str(dt[i].max()))>=0][0]
            #ref_max=pd.Series(vv[vv.isnull()!=True].max()).astype(str)[0]
            ref_min=pd.Series(vv[vv.isnull()!=True].min()).astype(str)[0]
            ref_min_mod=ref_min.replace(ref_min[ref_min.find('(')+1:ref_min.find(',')+1],'-inf,')
            ref_min_mod=ref_min_mod.replace(' ','')
            ref_max_mod=ref_max.replace(ref_max[ref_max.find(','):ref_max.find(']')],',inf')
            ref_max_mod=ref_max_mod.replace(' ','')
            vv=vv.replace({ref_min:ref_min_mod,ref_max:ref_max_mod})
            dt[i]=vv
        else:
            dt[i]=dt[i].astype('object')
    if len(feat)>0:
        np.save(path+'/cut_feat', feat)
    return dt
        


def interval(dt,n_bins=5,target=None):
    if target is None:
        ref_var=dt.dtypes
    else:
        ref_var=dt[dt.columns[dt.columns!=target]].dtypes
    for i in ref_var[ref_var!='object'].index:
        quantil=dt[i].quantile(np.arange(0,1+1/n_bins,1/n_bins))
        quantil=np.unique(quantil)
        if len(quantil)>2:
            vv=pd.cut(dt[i],quantil, include_lowest=True).astype(str)
            ref_max=pd.Series(vv[vv.isnull()!=True].max()).astype(str)[0]
            ref_min=pd.Series(vv[vv.isnull()!=True].min()).astype(str)[0]
            ref_min_mod=ref_min.replace(ref_min[ref_min.find('(')+1:ref_min.find(',')+1],'-inf,')
            ref_min_mod=ref_min_mod.replace(' ','')
            ref_max_mod=ref_max.replace(ref_max[ref_max.find(','):ref_max.find(']')],',inf')
            ref_max_mod=ref_max_mod.replace(' ','')
            vv=vv.replace({ref_min:ref_min_mod,ref_max:ref_max_mod})
            dt[i]=vv
        else:
            dt[i]=dt[i].astype('object')
    return dt

def importante_feature(dt,target):
    inf_g=[info_gain.info_gain(dt[i], dt[target]) for i in dt.columns[dt.columns!=target]]
    pv=pd.DataFrame({'features':dt.columns[dt.columns!=target],'GI':inf_g})
    pv.sort_values(by=['GI'],ascending=False,inplace=True)
    pv.reset_index(drop=True,inplace=True)
    pv.drop(index=list(pv.index)[pv.shape[0]-1],inplace=True)
    return pv

pv=importante_feature(dt,target)
dt_mod=dt.copy()
#dt_mod=interval(dt_mod,n_bins,target)
dt_mod=interval_woe(dt_mod,target,label,n_bins,path)

dt_mod['duration'].value_counts()

perc_ref=dt[target].value_counts(normalize=True)[label]

i='balance'
idx=0




#feature(dt_mod,j,i,target,label)
#dt=dt_mod;cond=j;var=i

def feature(dt,cond,var,target,label,index=None):
    if index is None:
        index=np.where(dt[var]==cond)[0]
    qtd=len(index)
    taxa_massa=qtd/dt.shape[0]
    calc_ref=dt.loc[index,target].value_counts(normalize=True)
    perc_ref=dt[target].value_counts(normalize=True)[label]
    if sum(calc_ref.index==label)>0:
        qtd_class_label=dt.loc[index,target].value_counts()[label]
        
    else:
        qtd_class_label=0
    
    pvalor=stat.stats.binom_test(qtd_class_label, qtd, perc_ref, alternative='two-sided')
    
     
    
    
    
    if sum(calc_ref.index==label)>0:
        lift=calc_ref[label]/perc_ref
        taxa=dt.loc[index,target].value_counts(normalize=True)[label]
        per_events=dt.loc[index,target].value_counts(normalize=True)[label]
        nper_events=1-per_events
        if qtd>1:
            z=np.sqrt((per_events*nper_events)/(qtd-1))
        else:
            z=np.sqrt((per_events*nper_events)/(qtd-1+0.001))
        woe=np.log(nper_events/per_events)
        iv=(nper_events-per_events)*woe
    else:
        lift=0
        taxa=0
        per_events=0
        nper_events=1
        woe=np.log((nper_events+0.01)/(per_events+0.01))
        if qtd>1:
            z=np.sqrt((per_events*nper_events)/(qtd-1))
        else:
            z=np.sqrt((per_events*nper_events)/(qtd-1+0.001))
        iv=(nper_events-per_events)*woe
    return pd.Series({'size':qtd,'lift':lift,'Taxa':taxa,'IV':iv,'Taxa_massa':taxa_massa,'p-valor':pvalor,'qtd_alvo':qtd_class_label,'Cobertura_Regra':z})

# x=j[0:j.find('==')-1]+' == '+j[j.find('==')+3:len(j)]
# x=j[0:j.find('==')-1]+' == '+j[j.find('==')+3:len(j)]+' & '+var_best_info+' == '+jj 
# x=j[0:j.find('==')-1]+' == '+j[j.find('==')+3:len(j)]+' & '+var_best_info+' == '+jj +' & '+var_best_info+' == '+jj

def generate_script(x,name_data):
    #name_data='dt_mod'
    start=[]
    end=[]
    for m in re.finditer('==', x):
        start.append(m.start())
        end.append(m.end())
    start_=[]
    end_=[]    
    for m in re.finditer('&', x):
        start_.append(m.start())
        end_.append(m.end())
        
    for m in range(len(start)):
        if (m==0) & (len(start_)==0):
            z='np.where(('+name_data+'["'+x[0:start[m]-1]+'"] == "'+ x[end[m]+1:len(x)]+ '"))[0]' 
        elif (m==0) & (len(start_)>0):
            z='np.where(('+name_data+'["'+x[0:start[m]-1]+'"] == "'+ x[end[m]+1:start_[m]-1]+'") & ('
        elif (m>0) & (len(start_)>m):
            z=z+name_data+'["'+x[end_[m-1]+1:start[m]-1]+'"]'+ ' == "'+x[end[m]+1:start_[m]-1]+'") & ('
        elif (m>0) & (len(start_)==m):
            z=z+name_data+'["'+x[end_[m-1]+1:start[m]-1]+'"]'+ ' == "'+x[end[m]+1:len(x)]+'"))[0]'
            
    return z

# x='balance == (1859.0, 102127.0]'
# var=var_best_info 
# level='(223.0, 368.0]'           
def create_rule(x,var,level):
    x=x+' & '+ var+' == '+level
    return x

#x=j[0:j.find('==')-1]+' == '+j[j.find('==')+3:len(j)]+' & '+var_best_info+' == '+jj +' & '+var_best_info+' == '+jj
def extract_features(x):
    start=[]
    end=[]
    for m in re.finditer('==', x):
        start.append(m.start())
        end.append(m.end())
    start_=[]
    end_=[]    
    for m in re.finditer('&', x):
        start_.append(m.start())
        end_.append(m.end())
    
    features=[]
    for m in range(len(start)):
        if (m==0):
            features.append(x[0:start[m]-1])
        else:
            features.append(x[end_[m-1]+1:start[m]-1])
    return features

#x=rules        
def unique_rules(x):
    index_level0=[]
    for ref,i in enumerate(x.Order.unique()):
        regras=x.iloc[np.where(x.Order==i)[0],:]
        size=[sum(np.array(list(j))=='&') for j in regras.Regras]
        regras['qtd_levels']=size
        regras.sort_values(by=['qtd_levels'],ascending=False,inplace=True)
        regras.reset_index(drop=True,inplace=True)
        unique_rule=[]
        unique_rule_mod=[]
        for j in regras.index:
            if j==0:
                index=list(eval(generate_script(regras.loc[j,:].Regras,'dt_mod')))
                unique_rule.append(1)
                if ref==0:
                    index_level0.extend(index)
                    
                    
            else:
                idx=eval(generate_script(regras.loc[j,:].Regras,'dt_mod'))    
                if len(np.where(np.isin(np.array(index),idx))[0])==0:
                    unique_rule.append(1)
                    if ref==0:
                        index_level0.extend(idx)
                else:
                    unique_rule.append(0)
                index.extend(list(idx))
                
            if ref==0:
                unique_rule_mod=[0]*regras.shape[0]
            else:
                index2=eval(generate_script(regras.loc[j,:].Regras,'dt_mod'))
                if len(np.where(np.isin(index2,np.array(index_level0)))[0])==0:
                    unique_rule_mod.append(1)
                else:
                    unique_rule_mod.append(0)
                    
        
        
        regras['unique_rule']=unique_rule
        #regras['unique_rule_mod']=unique_rule_mod
        
        if ref==0:
            regras_final=regras
        else:
            regras_final=pd.concat([regras_final,regras])
       
    regras_final=regras_final.drop(columns=['Level','IV'])             
    return regras_final

def unique_rule_all(x):
    index_level0=[]
    idx_ref=np.where((x['unique_rule']==1) & (x['Order']==x['Order'].min()))[0]
    for idx,i in enumerate(idx_ref):
        if idx==0:
            index=eval(generate_script(x.Regras[i],'dt_mod'))
            index_level0.extend(list(index))
        else:
            index_level0.extend(list(eval(generate_script(x.Regras[i],'dt_mod'))))
    
    for idx,i in enumerate(np.where(x['Order']>x['Order'].min())[0]):
        if idx==0:
            x.loc[idx_ref,'unique_rule_mod']=len(idx_ref)*[1]
        index=list(eval(generate_script(x.loc[i,:].Regras,'dt_mod')))
        if len(np.where(np.isin(np.array(index),index_level0))[0])==0:
            x.loc[i,'unique_rule_mod']=1
            index_level0.extend(list(index))
        else:
            x.loc[i,'unique_rule_mod']=0
    
    return x

# path_file=path+'/cut_feat.npy'
# x=dt.copy() 
# rules_select=rules.iloc[0:4,:]           
def pred(x,rules_select,path_file=None):
    x_mod=x.copy()
    cutx= np.load(path_file,allow_pickle=True).tolist()
    for i in cutx.keys():
        cutx[i][0]=x_mod[i].min()
        cutx[i][len(cutx[i])-1]=x_mod[i].max()
        
        vv=pd.cut(x[i],cutx[i], include_lowest=True).astype(str)
        ref_max=[j for j in vv.unique() if j.find(str(dt[i].max()))>=0][0]
        #ref_max=pd.Series(vv[vv.isnull()!=True].max()).astype(str)[0]
        ref_min=pd.Series(vv[vv.isnull()!=True].min()).astype(str)[0]
        ref_min_mod=ref_min.replace(ref_min[ref_min.find('(')+1:ref_min.find(',')+1],'-inf,')
        ref_min_mod=ref_min_mod.replace(' ','')
        ref_max_mod=ref_max.replace(ref_max[ref_max.find(','):ref_max.find(']')],',inf')
        ref_max_mod=ref_max_mod.replace(' ','')
        vv=vv.replace({ref_min:ref_min_mod,ref_max:ref_max_mod})
        x_mod[i]=vv
    for idx,i in enumerate(rules_select.Regras):
        index=eval(generate_script(i,'x'))
        values=np.array([0]*x.shape[0])
        values[index]=1
        x['regra_'+str(idx)]=values
    return x
    
    
        
        

rules=pd.DataFrame(columns=['Regras','Lift','Taxa_Massa','Taxa_Alvo','Qtd_Cubo','Qtd_Alvo_Cubo','IV','Level','Order','p-value','Cobertura_Regra'])
for idx,i in enumerate(tqdm(pv.features)):
    levels_feat=dt_mod[i].unique()

    for k1 in range(level_rule):
         
        if k1==0:
            for j in levels_feat:
                feat=feature(dt_mod,j,i,target,label)
                if (feat['size']>int(dt_mod.shape[0]*perc)): #int(dt_mod.shape[0]*perc)
                    rules = rules.append({'Regras': i+' == '+j,'Lift':feat['lift'],'Taxa_Massa':feat['Taxa_massa'],'Taxa_Alvo':feat['Taxa'],'IV':feat['IV'],'Level':i+'_'+str(k1),'Order':idx,'p-value':feat['p-valor'],'Qtd_Alvo_Cubo':feat['qtd_alvo'],'Qtd_Cubo':feat['size'],'Cobertura_Regra':feat['Cobertura_Regra']}, ignore_index=True)
        else:
            for j in rules[rules['Level']==i+'_'+str(k1-1)].Regras.values:
                index=eval(generate_script(j,'dt_mod'))
                not_feature=pv.features[0:(idx+1)].tolist()
                not_feature.append(target)
                not_feature.extend(extract_features(j))
                not_feature=list(np.unique(not_feature))
                feat_ref=dt.columns[~np.isin(dt.columns,np.array(not_feature))]
                if len(feat_ref)>0:
                    inf_g_filt=[info_gain.info_gain(dt.loc[index,k], dt.loc[index,target]) for k in feat_ref]
                    var_best_info=feat_ref[np.argmax(inf_g_filt)]
                   
                    for jj in dt_mod[var_best_info].unique():
                        rule_=create_rule(j,var_best_info,jj)
                        index=eval(generate_script(rule_,'dt_mod'))     
                        feat=feature(dt_mod,j,i,target,label,index)
                        if (feat['size']>int(dt_mod.shape[0]*perc)): #int(dt_mod.shape[0]*perc)
                            rules = rules.append({'Regras': rule_,'Lift':feat['lift'],'Taxa_Massa':feat['Taxa_massa'],'Taxa_Alvo':feat['Taxa'],'IV':feat['IV'],'Level':i+'_'+str(k1),'Order':idx,'p-value':feat['p-valor'],'Qtd_Alvo_Cubo':feat['qtd_alvo'],'Qtd_Cubo':feat['size'],'Cobertura_Regra':feat['Cobertura_Regra']}, ignore_index=True)


rules=rules.loc[np.where(rules['p-value']<=0.05)[0],:]
rules=rules.reset_index(drop=True)  
rules=unique_rules(rules) 
rules=rules.reset_index(drop=True) 
rules=unique_rule_all(rules)

x=rules
def unique_rule_all(x):
    index_level0=[]
    idx_ref=np.where((x['unique_rule']==1) & (x['Order']==x['Order'].min()))[0]
    for idx,i in enumerate(idx_ref):
        if idx==0:
            index=eval(generate_script(x.Regras[i],'dt_mod'))
            index_level0.extend(list(index))
        else:
            index_level0.extend(list(eval(generate_script(x.Regras[i],'dt_mod'))))
    
    for idx,i in enumerate(np.where(x['Order']>x['Order'].min())[0]):
        if idx==0:
            x.loc[idx_ref,'unique_rule_mod']=len(idx_ref)*[1]
        index=list(eval(generate_script(x.loc[i,:].Regras,'dt_mod')))
        if len(np.where(np.isin(np.array(index),index_level0))[0])==0:
            x.loc[i,'unique_rule_mod']=1
            index_level0.extend(list(index))
        else:
            x.loc[i,'unique_rule_mod']=0
    
    return x
        
            
        
        


index=[]
for idx,i in enumerate(np.where((rules['unique_rule_mod']==1))[0]): #& (rules['Order']==0)
    if idx==0:
        index.extend(list(eval(generate_script(rules.loc[i,'Regras'],'dt_mod'))))
    else:
        idxx=eval(generate_script(rules.loc[i,'Regras'],'dt_mod'))
        print(len(np.where(np.isin(np.array(index),idxx))[0]))
        index.extend(list(idxx))                   

                    
                
                
                
                   
                   

                
         



        
        








