from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
pd.set_option('display.max_colwidth',None)


# gather all in- and exclusion criteria in one dataframe
def gather_criteria(nodes, criteria=True):
    conditions = pd.DataFrame()
    for i in range(nodes.shape[0]):
        if isinstance(nodes.iloc[i,:]['conditions'],str):
            row=pd.DataFrame(columns=['feature','description','condition'])
            #row.loc[0,'conditions']=True
            #row.loc[0,'feature']= 'None'
        else:
            row=nodes.iloc[i,:]['conditions']
        row['cluster']=i
        conditions=pd.concat([conditions,row])
    # turn into matrix with one row per cluster, one column per disease group
    # which is 1 it the disase is included and 0 otherwise
    cond_binary = pd.DataFrame(0, index=list(range(0,132)),columns=list(range(0,131)))
    conditions['value'] = 1
    if criteria == True:
        cond_binary = (cond_binary+conditions[conditions['condition']==True][['feature','cluster','value']].pivot(columns='feature',index='cluster',values='value')).fillna(0)
    else:
        cond_binary = (cond_binary+conditions[conditions['condition']==False][['feature','cluster','value']].pivot(columns='feature',index='cluster',values='value')).fillna(0)
    cond_binary = cond_binary.fillna(0)
    return cond_binary


# calculate jaccard distance between two sets
def jaccard_distance(set1,set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    jaccard_dist = 1.0-(intersection/union) if union !=0 else 0.0
    return jaccard_dist


def jacc_dist_mod(conditions_false1, conditions_false2, conditions_true1, conditions_true2):
    '''
    conditions_false1, conditions_false2: matrix with dimensions (#clusters,#disease_blocks), with 1 if a specific disease block is an exclusion criterion in a cluster and 0 else
    conditions_true1, conditions_true2: matrix with dimensions (#clusters,#disease_blocks), with 1 if a specific disease block is an inclusion criterion in a cluster and 0 else

    The function calculates jaccard distance for inclusion criteria and exclusion criteria and then sums them up.
    '''
    # calculate modified jaccard distance between two vectors
    jacc_dist = np.empty((132,132))
    # compute similarity based on jaccard distance
    for i in range(132):
        for j in range(132):
            # calculate jaccard distance for false set
            false_set1 = set(conditions_false1.loc[i,conditions_false1.loc[i]==1].index)
            false_set2 = set(conditions_false2.loc[j,conditions_false2.loc[j]==1].index)
            false_dist = jaccard_distance(false_set1, false_set2)

            # calculate jaccard distance for true set
            true_set1 = set(conditions_true1.loc[i,conditions_true1.loc[i]==1].index)
            true_set2 = set(conditions_true2.loc[j,conditions_true2.loc[j]==1].index)
            true_dist = jaccard_distance(true_set1, true_set2)
            # sum them
            jacc_dist[i,j] = false_dist+ true_dist
    return(jacc_dist)

def acc_dist(conditions1,conditions2):
    '''
    conditions_false1, conditions_false2: matrix with dimensions (#clusters,#disease_blocks), with 1 if a specific disease block is an exclusion criterion in a cluster and 0 else
    conditions_true1, conditions_true2: matrix with dimensions (#clusters,#disease_blocks), with 1 if a specific disease block is an inclusion criterion in a cluster and 0 else

    The function calculates jaccard distance for inclusion criteria and exclusion criteria and then sums them up.
    '''
  # calculate modified jaccard distance between two vectors
    acc_dist = np.empty((132,132))
    # compute similarity based on jaccard distance
    for i in range(132):
        temp=0
        div = conditions1.shape[0]
        for j in range(132):
            try:
                temp += sum(np.logical_and(conditions1.loc[i,:],conditions2.loc[j,:])) / sum(np.logical_or(conditions1.loc[i,:],conditions2.loc[j,:]))
            except:
                div -= 1
        acc_dist[i,j]=temp/div
    return 1- acc_dist






# get names and description from blocks file
def get_block_info(df,blocks):
    '''
    df: a vector with length 131 that is True if we want the description of this specific condition and false otherwise

    blocks: file containing ICD 10 code diagnosis blocks descriptions
    '''
    df = blocks[df]
    df['ICD-10'] = df['from_ICD'] +'-'+ df['to_ICD']
    df  = df .drop(['diagnosis_ids','from_ICD','to_ICD','chapter'],axis=1)
    df = df.rename(columns = {'description':'Description'})[['ID','ICD-10','Description']]
    return(df)



def cluster_to_latex(path,df,cluster,blocks):
    conditions = df.loc[cluster]['conditions']
    conditions = conditions.sort_values('from_ICD')
    conditions = conditions.merge(blocks,how='left',left_on='ID',right_on='ID')
    tab = conditions['description_y']+' ('+conditions['from_ICD_y']+'-'+conditions['to_ICD_y']+')'
    tab = pd.DataFrame(tab).rename(columns={0:'Disease Blocks'})
    tab['Criteria'] = conditions['condition']

    tab.loc[tab['Criteria']==False,'Criteria']="\\"+"xmark"
    tab.loc[tab['Criteria']==True,'Criteria'] = """\checkmark"""

    # convert to latex
    latex_table = tab.to_latex(index=False, column_format='ll',bold_rows=False, caption ='Cluster '+str(cluster),escape=False)
    # column_widths = {'Disease Blocks':'8cm','Criteria':'2cm'}
    # for column, width in column_widths.items():
    #     latex_table = latex_table.replace(f'{{{column}}}',f'p{{{width}}}')
    # # center multicolumn
    # latex_table =latex_table.replace('{l}','{Y}')
    latex_content = r"""
    \begin{center}
    \captionsetup{justification=centering}

    %s
    \label{%s}
    \end{center}""" % (latex_table,'Cluster_'+str(cluster))

    # resize to fit to page
    latex_content = latex_content.replace(r"\begin{tabular}{ll}",r"\footnotesize \begin{tabularx}{\textwidth}{P{10cm}P{2cm}}")
    latex_content = latex_content.replace(r"end{tabular}",r"end{tabularx}")

    # maek sure appears after headline
    latex_content = latex_content.replace(r"\begin{table}",r"\begin{table}[htbp]")

    # add horizontal partial lines
    #latex_content = latex_content.replace(r"\begin{tabular}",r"\footnotesize \begin{tabularx}{12cm}")

    with open(os.path.join(path,'Cluster_'+str(cluster)+'.tex'),'w') as file:
        file.write(latex_content)   

# get difference tables in latex
def diff_to_latex(conditions_false1, conditions_false2, conditions_true1, conditions_true2, path,blocks,names=['Clustering 1','Clustering 2']):
    '''
    conditions_false1, conditions_false2: matrix with dimensions (#clusters,#disease_blocks), with 1 if a specific disease block is an exclusion criterion in a cluster and 0 else
    conditions_true1, conditions_true2: matrix with dimensions (#clusters,#disease_blocks), with 1 if a specific disease block is an inclusion criterion in a cluster and 0 else

    conditions1 and conditions2 should be matched according to sum distance measure and linear sum assignment
    '''
    for cluster in conditions_false1.index:
        #clsuter=0
        # get exclusion cireteria that are in the first clustering but not in the second
        ex_not_in_2 = get_block_info(((conditions_false1.iloc[cluster] - conditions_false2.iloc[cluster])==1),blocks)
        ex_not_in_2['Clustering 1']= "\\"+"xmark"
        ex_not_in_2['Clustering 2']= ""


        # get exclusion cireteria that are in the second clustering but not in the first
        ex_not_in_1 = get_block_info(((conditions_false1.iloc[cluster] - conditions_false2.iloc[cluster])==-1),blocks)
        ex_not_in_1['Clustering 1']= ""
        ex_not_in_1['Clustering 2']= "\\"+"xmark"

        ex = pd.concat([ex_not_in_2, ex_not_in_1])
        ex.set_index(['ID','ICD-10','Description'],inplace=True)


        # get inclusion cireteria that are in the first clustering but not in the second
        in_not_in_1 = get_block_info(((conditions_true1.iloc[cluster] - conditions_true2.iloc[cluster])==1),blocks)
        in_not_in_1['Clustering 1']= """\checkmark"""
        in_not_in_1['Clustering 2']= ""

        # get inclusion cirteria that are in the second clustering but not in the first
        in_not_in_2 =get_block_info(((conditions_true1.iloc[cluster] - conditions_true2.iloc[cluster])==-1),blocks)
        in_not_in_2['Clustering 2']= ""
        in_not_in_2['Clustering 2']= """\checkmark"""

        inc = pd.concat([in_not_in_1, in_not_in_2])
        inc.set_index(['ID','ICD-10','Description'],inplace=True)

        # merge both together
        df = ex.merge(inc,left_index=True, right_index=True,how='outer')
        df = df.fillna('')
        df['\shortstack{' +names[0]+' \\\Cluster '+str(conditions_false1.index[cluster])+'}'] = df['Clustering 1_x']+df['Clustering 1_y']
        df['\shortstack{' + names[1]+' \\\Cluster '+str(conditions_false2.index[cluster])+'}'] = df['Clustering 2_x']+df['Clustering 2_y']
        df = df.drop(['Clustering 1_x','Clustering 1_y','Clustering 2_x','Clustering 2_y'],axis=1)
        if df.shape[0]!=0:
            # convert to latex
            latex_table = df.to_latex(index=True, column_format='llXssss',bold_rows=False, caption ='Cluster '+str(cluster),escape=False)
            # center multicolumn
            latex_table =latex_table.replace('{l}','{c}')
            latex_content = r"""
            \begin{center}
            \captionsetup{justification=centering}
        
            %s
            \end{center}""" % latex_table

            # resize to fit to page
            latex_content = latex_content.replace(r"\begin{tabular}",r"\footnotesize \begin{tabularx}{12cm}")
            latex_content = latex_content.replace(r"end{tabular}",r"end{tabularx}")

            # maek sure appears after headline
            latex_content = latex_content.replace(r"\begin{table}",r"\begin{table}[h]")

            # add horizontal partial lines
            latex_content = latex_content.replace(r"\begin{tabular}",r"\footnotesize \begin{tabularx}{12cm}")

            with open(os.path.join(path,'Cluster_'+str(cluster)+'.tex'),'w') as file:
                file.write(latex_content)   

def Accuracy(y1,y2):
    ''' Function to compute accuracy, i.e. hamming score'''
    temp = 0
    div = y1.shape[1]
    for i in range(y1.shape[1]):
        try:
            temp += sum(np.logical_and(y1.loc[:,i],y2.loc[:,i])) / sum(np.logical_or(y1.loc[:,i],y2.loc[:,i]))
        except:
            div -= 1
    return temp/div

def get_confusion_matrices(conditions1, conditions2,path):
    '''
    conditions1, conditoins2: matrix with dimensions (#clusters,#disease_blocks), with 1 if a specific disease block is an inclusion criterion in a cluster, -1 if it is an exclusion criterion and 0 else

    conditions1 and conditions2 should be matched according to sum distance measure and linear sum assignment
    '''
    plt.rcParams['font.size'] = 7
    confusion_matrices = []
    vmax = 0
    for cluster in conditions1.index:
        cm = confusion_matrix(conditions1.loc[cluster],conditions2.loc[cluster], labels=[-1,0,1])
        off_diag_mask = np.eye(*cm.shape,dtype=bool)
        if np.sum(cm[~off_diag_mask])!=0:
            confusion_matrices.append([cluster,cm])
            if vmax<cm[~off_diag_mask].max():
                vmax = cm[~off_diag_mask].max()
    
    num_plots = len(confusion_matrices) 
    plots_per_page = 40
    num_cols = 5
    num_rows = int(np.ceil(plots_per_page/num_cols))
    num_pages = int(np.ceil(num_plots/plots_per_page))

    for page in range(num_pages):
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(1*num_cols,1*num_rows))
        for ax in axes:
            for sub_ax in ax:
                sub_ax.set_axis_off()

        plt.subplots_adjust(hspace=0.5,wspace=0)
        for i in range(plots_per_page):
            #print(cluster)
            #cm[~np.isnan(cm) * cm!=0] = cm[~np.isnan(cm) * cm!=0]-100
    
            #cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]*100
            ax = axes[i // num_cols, i % num_cols ] if num_rows >1 else axes[i % num_cols]   
            #ax = axes[i] if plots_per_page >1 else axes
            plot_num = page * plots_per_page + i + 1
            if plot_num <= num_plots:
                #print(plot_num)
                cm = confusion_matrices[plot_num-1][1]
                ax.set_axis_on()
                sns.heatmap(cm, annot = True, mask=off_diag_mask, fmt ='.0f',cbar=False, cmap='Blues', vmin=0, vmax=vmax, square=True, xticklabels = False, yticklabels=False,ax=ax, linecolor='black',linewidths=2)
                sns.heatmap(cm, annot = True, mask=~off_diag_mask, fmt ='.0f',cbar=False, cmap='Reds', annot_kws={'color':'black'}, alpha=0,square=True, xticklabels = False, yticklabels=False,ax=ax)
                ax.set_title('Cluster '+str(confusion_matrices[plot_num-1][0]))
                if ((plot_num-1) % plots_per_page)>(plots_per_page -num_cols-1):
                    ax.set_xticks([0.5,1.5,2.5],labels=['EX','A','IN'])
                if (plot_num-1) % num_cols == 0:
                    ax.set_yticks([0.5,1.5,2.5],labels=['EX','A','IN'])
    #plt.text(0.5, 0.97,'Austria', ha='center',fontsize=12, fontweight='bold')
    #plt.text(0.97, 0.5,'Denmark', ha='center',fontsize=12, fontweight='bold')

    #plt.tight_layout()
        plt.savefig(os.path.join(path,'confusion_matrices_p'+str(page)+'.pdf'))
        plt.close(fig)