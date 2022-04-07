import models
import pandas as pd

def get_representations():
    
    dic_tw = {'Binary'  : 0,
              'TF'      : 0,
              'TF-IDF'  : 0,
              }
    return dic_tw

def main():
    
    bow_s = get_representations()
    model = models.get_models()
    metric = "macro avg"
    error_type = "f1-score"
    per = float(0.3)
    commodity = ["soja", "milho"]
    #commodity = ["milho"]
    tipes = ["mul"]
    
    for cmt in commodity:
    
        for tip in tipes:
            
            if tip == 'bin' and per > 0:
                Columns = ['model', 'per', 'Binary', 'TF', 'TF-IDF', 'support0', 'support1']
            elif tip == 'bin' and per < 0:
                Columns = ['model', 'per', 'Binary', 'TF', 'TF-IDF', 'support-1', 'support0']
            elif tip == 'mul':
                Columns = ['model', 'per', 'Binary', 'TF', 'TF-IDF', 'support-1', 'support0', 'support1']
    
            df = pd.DataFrame(columns=Columns)

            for name_clf, classifier in model.items():
                aux_row = [name_clf, per]
                for n_vec, vectorizor in bow_s.items():
                    
                    file = str("res_base/"+str(cmt)+"_"+str(tip)+"_tit/"+str(cmt)+"_"+str(per)+"_"+str(n_vec)+"_"+str(name_clf)+".csv")
                    
                    if tip == "bin" and per > 0:
                        try:
                            df_aux = pd.read_csv(file, index_col=0)
                            
                            vl = df_aux.loc[ df_aux.index == metric, [error_type] ].values[0][0] #Binário
                            sp0 = df_aux.loc[ df_aux.index == '0', ['support'] ].values[0][0] #Binário
                            sp1 = df_aux.loc[ df_aux.index == '1', ['support'] ].values[0][0] #Binário
                            aux_row.append(float(round(vl, 5)))
                        except:
                            print("Ainda não processado: ", file.split('/')[2])
                            aux_row.append(0)
                            sp0, sp1 = 0, 0
                            
                    elif tip == "bin" and per < 0:
                        try:
                            df_aux = pd.read_csv(file, index_col=0)
                            vl = df_aux.loc[ df_aux.index == metric, [error_type] ].values[0][0] #Binário
                            sp_1 = df_aux.loc[ df_aux.index == '-1', ['support'] ].values[0][0] #Binário
                            sp0 = df_aux.loc[ df_aux.index == '0', ['support'] ].values[0][0] #Binário
                            aux_row.append(float(round(vl, 5)))
                        except:
                            print("Ainda não processado: ", file.split('/')[2])
                            aux_row.append(0)
                            sp_1, sp0 = 0, 0
                    
                    elif tip == 'mul':
                        try:
                            df_aux = pd.read_csv(file, index_col=0)
                            vl = df_aux.loc[ df_aux.index == metric, [error_type] ].values[0][0] #Binário
                            sp_1 = df_aux.loc[ df_aux.index == '-1', ['support'] ].values[0][0]
                            sp0 = df_aux.loc[ df_aux.index == '0', ['support'] ].values[0][0] 
                            sp1 = df_aux.loc[ df_aux.index == '1', ['support'] ].values[0][0] 
                            aux_row.append(float(round(vl, 5)))
                        except:
                            print("Ainda não processado: ", file.split('/')[2])
                            aux_row.append(0)
                            sp_1, sp0, sp1 = 0, 0, 0
                        
                if tip == 'bin' and per > 0:
                    aux_row.append(sp0)
                    aux_row.append(sp1)
                elif tip == 'bin' and per < 0:
                    aux_row.append(sp_1)
                    aux_row.append(sp0)
                elif tip == 'mul':
                    aux_row.append(sp_1)
                    aux_row.append(sp0)
                    aux_row.append(sp1)
        
                #print(aux_row)
                df.loc[len(df)] = aux_row
            
            df.to_excel("res_base/"+cmt+"_"+tip+"_"+str(per)+".xlsx")
         
if __name__ == '__main__':
    main()