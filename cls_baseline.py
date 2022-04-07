import models
import pandas as pd
import numpy as np
import re
import nltk
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit
from bertVectorizer import bertVectorizer
import sys
import nltk
nltk.download('rslp')

warnings.filterwarnings("ignore")
nltk.download('stopwords') 
nltk.download('punkt')

stop_words = nltk.corpus.stopwords.words('portuguese') 
stemmer = nltk.stem.RSLPStemmer()

def get_diff(df, column):
    df = df[column].diff().round(decimals=2)
    return df

def get_per(df, column):
    per = df[column].pct_change().round(decimals=4)
    return per

def join_texts(fontes, commoditie):
    
    texts = pd.DataFrame(columns=['titulo', 'conteudo', 'fonte'])
    
    for hd in fontes:
        tx = pd.read_csv('texts/'+hd+'_'+commoditie+'.csv', index_col='data')
        tx.index = pd.to_datetime(tx.index)
        tx = tx[['titulo', 'conteudo']]
        tx['fonte'] = str(hd)
        texts = texts.append(tx)   
    texts.sort_index(ascending=True, inplace=True) # sort df's by date(index)

    return texts

def add_target_texts(df_t, df_p, per, stp):
      
    if stp != 0:
        df_p['per_diff'] = df_p.per_diff.shift(periods=stp, fill_value=0)

    df = pd.merge(df_t, df_p['per_diff'], left_index=True, right_index=True)
    df['per_diff'] = df['per_diff'] * 100
    df['label'] = 0
    
    for p in per:
        if p > 0:
            df.loc[df['per_diff'] > p, 'label'] = 1
        elif p < 0:
            df.loc[df['per_diff'] < p, 'label'] = -1
    
    print("Number of Labels:\n",df['label'].value_counts())
    
    return df

def get_representations():
    
    dic_tw = {'Binary'  : CountVectorizer(stop_words=stop_words, binary=True),
              'TF'      : CountVectorizer(stop_words=stop_words),
              'TF-IDF'  : TfidfVectorizer(stop_words=stop_words),
              }
    
    return dic_tw

def get_classification_report(y_test, y_pred):
    report = classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    #df_classification_report = df_classification_report.sort_values(by=['f1-score'], ascending=False)
    aux = df.values.tolist()
    return aux

def clean_text(data):
    
    dt = []
    for row in range(0, len(data)):
      aux = ''
      text_top = str(data.iloc[row,]) 
      top_sub = re.sub('[0-9][^w]', '' , text_top)
      top_sub = re.sub(r'\n', '' , top_sub)
      aux += ' ' + str(top_sub)
      dt.append(aux)     
      
    return dt

def steming(dt):
    dt_st = []
    for st in dt:
        tx_st = [stemmer.stem(a) for a in st.split()]
        s = " "
        for tk in tx_st:
            s+= tk+" "
        dt_st.append(s.strip())
        
    return dt_st

def main():
    
    commodity = str(sys.argv[1])        #commodity - 'soja' or 'milho'
    tip = str(sys.argv[2])              #classification binary or multiclass - 'bin' o 'mult'
    per = float(sys.argv[3])              #percentage range - add label texts
    
    res_columns = ['precision', 'recall', 'f1-score', 'support']
    n_spl = 8                           #number of splits (tests = n_spl - 1)
    y_col = 'label'                     #column dependente variable
    X_col = 'titulo'                    #column text - 'titulo' or 'conteudo'
    
    df_price = pd.read_excel('prices/Cepea_'+str(commodity)+'.xls') # Load commoditie price file
    df_price['per_diff'] = get_per(df_price, 'real') # Get intraday price difference (percentage)
    df_price.set_index('date', inplace=True)
    df_tx = join_texts(['NA'], commodity) #3 texts sources - commodity name   

    if tip =='bin' and per > 0:
        df = add_target_texts(df_tx, df_price, [per], 0)
        res_ind = [0, 1, 'accuracy', 'macro avg', 'weighted avg']
    elif tip =='bin' and per < 0:
        df = add_target_texts(df_tx, df_price, [per], 0)
        res_ind = [-1, 0, 'accuracy', 'macro avg', 'weighted avg']
    elif tip == 'mul':
        df = add_target_texts(df_tx, df_price, [per, -per], 0)
        res_ind = [-1, 0, 1, 'accuracy', 'macro avg', 'weighted avg']
         
    mdls = models.get_models()
    bow_s = get_representations()
        
    for n_vec, vectorizor in bow_s.items():
        
        if  n_vec == 'TF-Bert' or n_vec == 'TF-Bert-Bigram':
            txts = pd.DataFrame()
            txt_aux = clean_text(df[X_col])
            txts['text'] = txt_aux
            len_test = int(txts.shape[0] / n_spl)    
        else:
            txt = clean_text(df[X_col])
            txts = steming(txt)
            len_test = int(len(txts) / n_spl)
            #print(len_test)
        
        for name_clf, classifier in mdls.items():
            
            print("Commodity: ", commodity, " - Per: " , per, " - BoW: ", n_vec, " - Classifier:" , name_clf)
            tscv = TimeSeriesSplit(n_splits = n_spl-1, test_size=len_test)
            
            res_spl = []
            y = df[y_col].values
            
            for train_index, test_index in tscv.split(txts, y):
                
                if n_vec == 'TF-Bert' or n_vec == 'TF-Bert-Bigram':
                    txts_spl = txts.iloc[0: test_index[-1] + 1]
                    matrix = vectorizor.fit_transform(txts_spl)
                    X = matrix.values
                else:
                     txts_spl = txts[0: test_index[-1] + 1]
                     #print(train_index[0:4], '...', train_index[-5:], "---", test_index[0:4], '...', test_index[-4:],)
                     matrix = vectorizor.fit_transform(txts_spl)
                     #XX = pd.DataFrame(matrix.todense(), columns=vect.get_feature_names(), index=df.index)                
                     XX = pd.DataFrame(matrix.todense(), columns=vectorizor.get_feature_names())
                     X = XX.values
                
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

            df_res.to_csv('res_'+str(commodity)+'_'+str(tip)+'_tit/'+str(commodity)+"_"+str(per)+"_"+str(n_vec)+"_"+str(name_clf)+".csv")
            
       
if __name__ == '__main__':
    main()
    

