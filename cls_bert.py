import models
import pandas as pd
import numpy as np
import re
import warnings
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit
from bertVectorizer import bertVectorizer
from sentence_transformers import SentenceTransformer
import sys
import warnings
warnings.filterwarnings("ignore")

def get_diff(df, column):
    df = df[column].diff().round(decimals=2)
    return df

def get_per(df, column):
    per = df[column].pct_change().round(decimals=4)
    return per

def add_target_texts(df_t, df_p, per, stp):
      
    if stp != 0:
        df_p['per_diff'] = df_p.per_diff.shift(periods=stp, fill_value=0)

    df = pd.merge(df_t, df_p['per_diff'], left_index=True, right_index=True)
    df['per_diff'] = df['per_diff'] * 100
    df['class'] = 'neu'
    
    for p in per:
        if p > 0:
            df.loc[df['per_diff'] > p, 'class'] = 'pos'
        elif p < 0:
            df.loc[df['per_diff'] < p, 'class'] = 'neg'
    
    print("Number of Class:\n",df['class'].value_counts())
    
    return df

def get_representations():
    
    dic_tw = {'Bert-Base'     : 'bert-base-multilingual-cased',
              'Distilbert'    : 'distilbert-base-multilingual-cased',
              'Bert-PtBr'     : 'neuralmind/bert-base-portuguese-cased',
              'TF-Bert'       : bertVectorizer(bert_model = 'bert-base-multilingual-cased', spacy_lang = 'pt_core_news_sm', lang='portuguese'),
              'TF-Distilbert' : bertVectorizer(bert_model = 'distilbert-base-multilingual-cased', spacy_lang = 'pt_core_news_sm', lang='portuguese'),
              'TF-Bert-PtBR'  : bertVectorizer(bert_model = 'neuralmind/bert-base-portuguese-cased', spacy_lang = 'pt_core_news_sm', lang='portuguese'),
              }
    
    return dic_tw

def get_classification_report(y_test, y_pred):
    report = classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    #df_classification_report = df_classification_report.sort_values(by=['f1-score'], ascending=False)
    aux = df.values.tolist()
    return aux

def main():
    
    commodity   = str(sys.argv[1])        #commodity - 'soja' or 'milho'
    tip         = str(sys.argv[2])        #classification binary or multiclass - 'bin' o 'mult'
    per         = float(sys.argv[3])      #percentage range - add label texts
    
    #commodity = 'soja'
    #tip = 'bin'
    #per = 0.3
    
    res_columns = ['precision', 'recall', 'f1-score', 'support']
    y_col = 'class'                     #column dependente variable
    X_col = 'titulo'                    #column text - 'titulo' or 'conteudo'
    n_spl = 8                           #number of splits (tests = n_spl - 1)
    
    df_price = pd.read_excel('prices/Cepea_'+str(commodity)+'.xls') # Load commoditie price file
    df_price['per_diff'] = get_per(df_price, 'real') # Get intraday price difference (percentage)
    df_price.set_index('date', inplace=True)
    
    df_tx = pd.read_csv('texts/NA_'+commodity+'.csv', index_col='data')
    df_tx.index = pd.to_datetime(df_tx.index)
    df_tx = df_tx[['titulo', 'conteudo']]
    df_tx.sort_index(ascending=True, inplace=True)
    
    if tip =='bin' and per > 0:
        df = add_target_texts(df_tx, df_price, [per], 0)
        res_ind = ['neu', 'pos', 'accuracy', 'macro avg', 'weighted avg']
    elif tip =='bin' and per < 0:
        df = add_target_texts(df_tx, df_price, [per], 0)
        res_ind = ['neg', 'neu', 'accuracy', 'macro avg', 'weighted avg']
    elif tip == 'mul':
        df = add_target_texts(df_tx, df_price, [per, -per], 0)
        res_ind = ['neg', 'neu', 'pos', 'accuracy', 'macro avg', 'weighted avg']
       
    mdls = models.get_models()
    bow_s = get_representations()
    
    for n_vec, vectorizor in bow_s.items():
        
        if  n_vec == 'TF-Bert' or n_vec == 'TF-Distilbert' or  n_vec == 'TF-Bert-PtBR':
            txts = pd.DataFrame()
            txts['text'] = df[X_col].to_list()
            len_test = int(txts.shape[0] / n_spl)  
        else:
            txts = df[X_col].to_list()
            len_test = int(len(txts) / n_spl)
            print(len(txts))
        
        for name_clf, classifier in mdls.items():
            
            print("Commodity: ", commodity, " - Per: " , per, " - Representation: ", n_vec, " - Classifier:" , name_clf)
            tscv = TimeSeriesSplit(n_splits = n_spl-1, test_size=len_test)
            
            res_spl = []
            y = df[y_col].values
            
            for train_index, test_index in tscv.split(txts, y):
                
                if n_vec == 'TF-Bert' or n_vec == 'TF-Distilbert' or  n_vec == 'TF-Bert-PtBR':
                    txts_spl = txts.iloc[0: test_index[-1] + 1]
                    matrix = vectorizor.fit_transform(txts_spl)
                    X = matrix.values
                else:
                    txts_spl = txts[0: test_index[-1] + 1]
                    model = SentenceTransformer(vectorizor)
                    X = model.encode(txts_spl)
                     
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                try:
                    clf = classifier
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    res_spl.append(get_classification_report(y_test, y_pred))
                    #print(get_classification_report(y_test, y_pred))
                except:
                    print("Erro:", n_vec, name_clf)
            
            res = np.array(res_spl).sum(axis=0)
            #res_aux = np.sum(res_med, axis=0)
            res[:, 0:3] = res[:, 0:3] / (n_spl - 1)
                        
            df_res = pd.DataFrame(res, columns=res_columns, index=res_ind)
            #print(df_res)

            df_res.to_csv('res_bert_'+str(commodity)+'/'+str(commodity)+'_'+str(tip)+'_'+str(per)+'_'+str(n_vec)+'_'+str(name_clf)+'.csv')
       
if __name__ == '__main__':
    main()
    
