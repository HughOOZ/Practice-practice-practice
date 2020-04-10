#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
from PIL import  Image
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import seaborn as sns
import itertools
import warnings
warnings.filterwarnings("ignore")
import io
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff


# In[24]:


data = pd.read_csv('C:/Users/HughOOZ/Downloads/WA_Fn-UseC_-HR-Employee-Attrition.csv')
data.head()


# In[25]:


print('row:',data.shape[0])
print('columns:',data.shape[1])
print('\nfeatures:',data.columns.tolist())
print('\nmissing values:',data.isnull().sum().values.sum())
print('\nunique values:','\n',data.nunique())


# In[26]:


data['BusinessTravel'] = data['BusinessTravel'].replace({'Non-Travel':0,'Travel_Rarely':1,'Travel_Frequently':2})


# In[35]:


attrition = data[data['Attrition'] == 'Yes']
not_attrition = data[data['Attrition'] == 'No']
data['ID'] = pd.Series(range(data.shape[0]))
Id_col = ['ID']
label_col = ['Attrition']
cat_cols = data.nunique()[data.nunique() < 6].keys().tolist()
cat_cols = [x for x in cat_cols if x not in label_col]
num_cols = [x for x in data.columns if x not in cat_cols + label_col + Id_col]
print(cat_cols,num_cols)


# In[51]:


lab = data['Attrition'].value_counts().keys().tolist()
val = data['Attrition'].value_counts().values.tolist()
trace = go.Pie(labels = lab ,
               values = val ,
               marker = dict(colors =  [ 'royalblue' ,'lime'],
                             line = dict(color = "white",
                                         width =  1.3)
                            ),
               rotation = 90,
               hoverinfo = "label+value+text",
               hole = .5
              )
layout = go.Layout(dict(title = "Employee Attrition in data",
                        plot_bgcolor  = "rgb(243,243,243)",
                        paper_bgcolor = "rgb(243,243,243)",
                       )
                  )
fig = go.Figure(data = [trace],layout = layout)
py.iplot(fig)


# In[53]:


def plot_pie(column) :
    
    trace1 = go.Pie(values  = attrition[column].value_counts().values.tolist(),
                    labels  = not_attrition[column].value_counts().keys().tolist(),
                    hoverinfo = "label+percent+name",
                    domain  = dict(x = [0,.48]),
                    name    = "attrition",
                    marker  = dict(line = dict(width = 2,
                                               color = "rgb(243,243,243)")
                                  ),
                    hole    = .6
                   )
    trace2 = go.Pie(values  = not_attrition[column].value_counts().values.tolist(),
                    labels  = not_attrition[column].value_counts().keys().tolist(),
                    hoverinfo = "label+percent+name",
                    marker  = dict(line = dict(width = 2,
                                               color = "rgb(243,243,243)")
                                  ),
                    domain  = dict(x = [.52,1]),
                    hole    = .6,
                    name    = "not attrition" 
                   )


    layout = go.Layout(dict(title = column + " distribution in attrition ",
                            plot_bgcolor  = "rgb(243,243,243)",
                            paper_bgcolor = "rgb(243,243,243)",
                            annotations = [dict(text = "attrition",
                                                font = dict(size = 13),
                                                showarrow = False,
                                                x = .15, y = .5),
                                           dict(text = "not attrition",
                                                font = dict(size = 13),
                                                showarrow = False,
                                                x = .88,y = .5
                                               )
                                          ]
                           )
                      )
    data = [trace1,trace2]
    fig  = go.Figure(data = data,layout = layout)
    py.iplot(fig)


def histogram(column) :
    trace1 = go.Histogram(x  = attrition[column],
                          histnorm= "percent",
                          name = "attrition",
                          marker = dict(line = dict(width = .5,
                                                    color = "black"
                                                    )
                                        ),
                         opacity = .9 
                         ) 
    
    trace2 = go.Histogram(x  = not_attrition[column],
                          histnorm = "percent",
                          name = "Not attrition",
                          marker = dict(line = dict(width = .5,
                                              color = "black"
                                             )
                                 ),
                          opacity = .9
                         )
    
    data = [trace1,trace2]
    layout = go.Layout(dict(title =column + " distribution in attrition ",
                            plot_bgcolor  = "rgb(243,243,243)",
                            paper_bgcolor = "rgb(243,243,243)",
                            xaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                             title = column,
                                             zerolinewidth=1,
                                             ticklen=5,
                                             gridwidth=2
                                            ),
                            yaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                             title = "percent",
                                             zerolinewidth=1,
                                             ticklen=5,
                                             gridwidth=2
                                            ),
                           )
                      )
    fig  = go.Figure(data=data,layout=layout)
    
    py.iplot(fig)
    


#for all categorical columns plot pie
for i in cat_cols :
    plot_pie(i)

#for all categorical columns plot histogram    
for i in num_cols :
    histogram(i)

#scatter plot matrix
#scatter_matrix(telcom)


# In[ ]:





# In[ ]:





# In[ ]:




