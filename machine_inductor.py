# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 14:17:36 2022

@author: valteresj
"""
import pandas as pd
import numpy as np
from info_gain import info_gain
import re
from tqdm import tqdm
from optbinning import OptimalBinning
import scipy as stat



def interval_woe(dt,target,label,n_bins=5,path=None,name_project=None):
    
    
    ref_var=dt[dt.columns[dt.columns!=target]].dtypes
    y = np.where(dt[target]==label,1,0)
    feat={}
    for i in tqdm(ref_var[ref_var!='object'].index):
        optb = OptimalBinning(name=i, dtype="numerical", solver="cp",min_n_bins=3,max_n_bins=n_bins)
        optb.fit(dt[i].values, y)
        cut=list(np.unique(optb.splits))
        cut.insert(0,dt[i].min())
        cut.append(dt[i].max())
        if len(cut)>2:
            feat[i]=cut
            vv=pd.cut(dt[i],cut, include_lowest=True,precision=4).astype(str)
            ref_max=[j for j in vv.unique() if j.find(str(round(dt[i].max(),4)))>=0][0]
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
        np.save(path+'/'+name_project+'.npy', feat)
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


def generate_script(x,name_data):
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


def create_rule(x,var,level):
    x=x+' & '+ var+' == '+level
    return x


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

       
def unique_rules(x,dt_mod):
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


def unique_rule_all(x,dt_mod):
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

def break_columns(x):
    for i in range(x.shape[0]):
        if x.loc[i,'Regras'].find('&')>0:
            start_=[]
            qtd=len([pos for pos, char in enumerate(x.loc[i,'Regras']) if char == '&'])
            for idx,j in enumerate(re.finditer('&', x.loc[i,'Regras'])):
                if idx==0:
                    start_.append(j.end())
                    
                    x.loc[i,'Clausula '+str(idx+1)]=x.loc[i,'Regras'][0:j.start()-1].replace('==','=')
                if (idx>0) & (idx<qtd):
                    
                    x.loc[i,'Clausula '+str(idx+1)]=x.loc[i,'Regras'][start_[idx-1]+1:j.end()-1].replace('==','=')
                    start_.append(j.end())
                if idx+1==qtd:
                    x.loc[i,'Clausula '+str(idx+2)]=x.loc[i,'Regras'][j.end()+1:len(x.loc[i,'Regras'])].replace('==','=')
        else:
            
            x.loc[i,'Clausula '+str(1)]=x.loc[i,'Regras'].replace('==','=')
    x.drop(columns=['Regras'],inplace=True)
    return x
    


class inductor(object):
    
    def __init__(self,dt=None,target=None,n_bins=5,label=None,level_rule=3,perc=0.01,t_lift=None,list_features=None,path=None,name_project=None,p_valor=0.05):           
        self.target=target
        self.n_bins=n_bins
        self.label=label
        self.level_rule=level_rule
        self.perc=perc
        self.t_lift=t_lift
        self.list_features=list_features
        self.path=path
        self.dt=dt
        self.name_project=name_project
        self.p_valor=p_valor
        
    def inductor_machine(self):
        pv=importante_feature(self.dt,self.target)
        dt_mod=self.dt.copy()
        dt_mod=interval_woe(dt_mod,self.target,self.label,self.n_bins,self.path,self.name_project)
        rules=pd.DataFrame(columns=['Regras','Lift','Taxa_Massa','Taxa_Alvo','Qtd_Cubo','Qtd_Alvo_Cubo','IV','Level','Order','p-value','Cobertura_Regra'])
        for idx,i in enumerate(tqdm(pv.features)):
            levels_feat=dt_mod[i].unique()

            for k1 in range(self.level_rule):
                 
                if k1==0:
                    for j in levels_feat:
                        feat=feature(dt_mod,j,i,self.target,self.label)
                        if (feat['size']>int(dt_mod.shape[0]*self.perc)) & ((feat['Taxa']>0) | (feat['Taxa']<1) ): 
                           rules = rules.append({'Regras': i+' == '+j,'Lift':feat['lift'],'Taxa_Massa':feat['Taxa_massa'],'Taxa_Alvo':feat['Taxa'],'IV':feat['IV'],'Level':i+'_'+str(k1),'Order':idx,'p-value':feat['p-valor'],'Qtd_Alvo_Cubo':feat['qtd_alvo'],'Qtd_Cubo':feat['size'],'Cobertura_Regra':feat['Cobertura_Regra']}, ignore_index=True)
                else:
                    for j in rules[rules['Level']==i+'_'+str(k1-1)].Regras.values:
                        index=eval(generate_script(j,'dt_mod'))
                        not_feature=pv.features[0:(idx+1)].tolist()
                        not_feature.append(self.target)
                        not_feature.extend(extract_features(j))
                        not_feature=list(np.unique(not_feature))
                        feat_ref=self.dt.columns[~np.isin(self.dt.columns,np.array(not_feature))]
                        if len(feat_ref)>0:
                            inf_g_filt=[info_gain.info_gain(self.dt.loc[index,k], self.dt.loc[index,self.target]) for k in feat_ref]
                            var_best_info=feat_ref[np.argmax(inf_g_filt)]
                           
                            for jj in dt_mod[var_best_info].unique():                               
                                rule_=create_rule(j,var_best_info,jj)
                                index=eval(generate_script(rule_,'dt_mod'))     
                                feat=feature(dt_mod,j,i,self.target,self.label,index)
                                if (feat['size']>int(dt_mod.shape[0]*self.perc)) & ((feat['Taxa']>0) | (feat['Taxa']<1) ): # & ((feat['p-valor']<=self.p_valor) | (feat['p-valor']>=(1-self.p_valor)) )
                                    rules = rules.append({'Regras': rule_,'Lift':feat['lift'],'Taxa_Massa':feat['Taxa_massa'],'Taxa_Alvo':feat['Taxa'],'IV':feat['IV'],'Level':i+'_'+str(k1),'Order':idx,'p-value':feat['p-valor'],'Qtd_Alvo_Cubo':feat['qtd_alvo'],'Qtd_Cubo':feat['size'],'Cobertura_Regra':feat['Cobertura_Regra']}, ignore_index=True)
                                    
        rules=rules.loc[np.where(( rules['p-value']<=0.05 ) & ((rules['Lift']>=1.2) | (rules['Lift']<=0.8)))[0],:]
        #rules=rules.loc[np.where( (rules['p-value']<=0.05)  & (rules['Lift']>=1.2))[0],:]
        rules=rules.reset_index(drop=True)                            
        rules=unique_rules(rules,dt_mod) 
        # rules=rules.reset_index(drop=True)
        # rules=unique_rule_all(rules,dt_mod)
        # rules=rules[rules['unique_rule_mod']==1]
        rules=rules.reset_index(drop=True)
        
        rules=rules.rename(columns={
            'Order':'Numero de Ordem',
            'qtd_levels':'Numero de Clausulas',
            'Qtd_Alvo_Cubo':'Numero de exemplos Alvo no cubo',
            'Qtd_Cubo':'Numero de exemplos no cubo',
            'Taxa_Massa':'Cobertura da Regra',
            'Taxa_Alvo':'Confianca da Regra',
            'Lift':'Lift da Regra',
            'p-value':'Significancia Estatistica da Regra'
            })

        rules=rules.loc[:,['Numero de Ordem',
        'Numero de Clausulas',
        'Numero de exemplos Alvo no cubo',
        'Numero de exemplos no cubo',
        'Cobertura da Regra',
        'Confianca da Regra',
        'Lift da Regra',
        'Significancia Estatistica da Regra','Regras']]
        
        rules=break_columns(rules)
        
        return rules
        
    def pred(self,new_data,rules_select):
        x_mod=new_data.copy()
        
        cutx= np.load(self.path+'/'+self.name_project+'.npy',allow_pickle=True).tolist()
        for i in cutx.keys():
            cutx[i][0]=x_mod[i].min()
            cutx[i][len(cutx[i])-1]=x_mod[i].max()
            
            vv=pd.cut(x_mod[i],cutx[i], include_lowest=True).astype(str)
            ref_max=[j for j in vv.unique() if j.find(str(x_mod[i].max()))>=0][0]
            #ref_max=pd.Series(vv[vv.isnull()!=True].max()).astype(str)[0]
            ref_min=pd.Series(vv[vv.isnull()!=True].min()).astype(str)[0]
            ref_min_mod=ref_min.replace(ref_min[ref_min.find('(')+1:ref_min.find(',')+1],'-inf,')
            ref_min_mod=ref_min_mod.replace(' ','')
            ref_max_mod=ref_max.replace(ref_max[ref_max.find(','):ref_max.find(']')],',inf')
            ref_max_mod=ref_max_mod.replace(' ','')
            vv=vv.replace({ref_min:ref_min_mod,ref_max:ref_max_mod})
            x_mod[i]=vv
        for idx,i in enumerate(rules_select.Regras):
            index=eval(generate_script(i,'self.dt'))
            values=np.array([0]*x_mod.shape[0])
            values[index]=1
            new_data['regra_'+str(idx)]=values
        return new_data
