# ## Data loading and cleaning
# Below are functions to load the dataset of your choice. After that, it is all up to you to create and evaluate a classification method. Beware, there may be missing values in these datasets. Good luck!


#%% Data loading functions. Uncomment the one you want to use
#from worcgist.load_data import load_data
#from worclipo.load_data import load_data
#from worcliver.load_data import load_data
#from hn.load_data import load_data
from ecg.load_data import load_data

import pandas as pd

data = load_data()
print(f'The number of samples: {len(data.index)}')

print(f'The number of columns: {len(data.columns)}')

data = pd.read_csv('ecg\ecg_data\ecg_data.csv')

#%%
# print(sum(data.iloc[:,-1:]))
print(data['label'].sum())

# Er zijn 146 gezonde patienten en 691 ongezonde patienten
