#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('laptop_data.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.duplicated().sum()


# In[7]:


df.isnull().sum()


# In[8]:


df.drop(columns=['Unnamed: 0'],inplace=True)


# In[9]:


df.head()


# In[10]:


df['Ram'] = df['Ram'].str.replace('GB','')
df['Weight'] = df['Weight'].str.replace('kg','')


# In[11]:


df.head()


# In[12]:


df['Ram'].astype('int32')
df['Weight']= df['Weight'].astype('float32')


# In[13]:


df.info()


# In[14]:


import seaborn as sns


# In[15]:


sns.distplot(df['Price'])


# In[16]:


df['Company'].value_counts().plot(kind='bar')


# In[17]:


sns.barplot(x=df['Company'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[18]:


df['TypeName'].value_counts().plot(kind='bar')


# In[19]:


sns.barplot(x=df['TypeName'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[20]:


sns.distplot(df['Inches'])


# In[21]:


sns.scatterplot(x=df['Inches'],y=df['Price'])


# In[22]:


df['ScreenResolution'].value_counts()


# In[23]:


df['Touchscreen'] = df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)


# In[24]:


df.sample(5)


# In[25]:


df['Touchscreen'].value_counts().plot(kind='bar')


# In[26]:


sns.barplot(x=df['Touchscreen'],y=df['Price'])


# In[27]:


df['Ips'] = df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)


# In[28]:


df.head()


# In[29]:


df['Ips'].value_counts().plot(kind='bar')


# In[30]:


sns.barplot(x=df['Ips'],y=df['Price'])


# In[31]:


new = df['ScreenResolution'].str.split('x',n=1,expand=True)


# In[32]:


df['X_res'] = new[0]
df['Y_res'] = new[1]
df.sample(5)


# In[33]:


df['X_res'] = df['X_res'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])


# In[34]:


df.head()


# In[35]:


df['X_res'] = df['X_res'].astype('int')
df['Y_res'] = df['Y_res'].astype('int')


# In[36]:


df.info()


# In[39]:


df.corr()['Price']


# In[38]:


df['ppi'] = (((df['X_res']**2) + (df['Y_res']**2))**0.5/df['Inches']).astype('float')


# In[40]:


df.corr()['Price']


# In[41]:


df.drop(columns=['ScreenResolution'],inplace=True)


# In[42]:


df.head()


# In[43]:


df.drop(columns=['Inches','X_res','Y_res'],inplace=True)


# In[44]:


df.head()


# In[45]:


df['Cpu'].value_counts()


# In[46]:


df['Cpu Name'] = df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))


# In[47]:


df.head()


# In[48]:


def fetch_processor(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'


# In[49]:


df['Cpu brand'] = df['Cpu Name'].apply(fetch_processor)


# In[50]:


df.head()


# In[51]:


df['Cpu brand'].value_counts().plot(kind='bar')


# In[52]:


sns.barplot(x=df['Cpu brand'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[53]:


df.drop(columns=['Cpu','Cpu Name'],inplace=True)


# In[59]:


df.head()


# In[60]:


df['Ram'].value_counts().plot(kind='bar')


# In[61]:


sns.barplot(x=df['Ram'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[62]:


df['Memory'].value_counts()


# In[63]:


df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)
df["Memory"] = df["Memory"].str.replace('GB', '')
df["Memory"] = df["Memory"].str.replace('TB', '000')
new = df["Memory"].str.split("+", n = 1, expand = True)

df["first"]= new[0]
df["first"]=df["first"].str.strip()

df["second"]= new[1]

df["Layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer1Hybrid"] = df["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer1Flash_Storage"] = df["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['first'] = df['first'].str.replace(r'\D', '')

df["second"].fillna("0", inplace = True)

df["Layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer2Hybrid"] = df["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer2Flash_Storage"] = df["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['second'] = df['second'].str.replace(r'\D', '')

df["first"] = df["first"].astype(int)
df["second"] = df["second"].astype(int)

df["HDD"]=(df["first"]*df["Layer1HDD"]+df["second"]*df["Layer2HDD"])
df["SSD"]=(df["first"]*df["Layer1SSD"]+df["second"]*df["Layer2SSD"])
df["Hybrid"]=(df["first"]*df["Layer1Hybrid"]+df["second"]*df["Layer2Hybrid"])
df["Flash_Storage"]=(df["first"]*df["Layer1Flash_Storage"]+df["second"]*df["Layer2Flash_Storage"])

df.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
       'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
       'Layer2Flash_Storage'],inplace=True)


# In[64]:


df.sample(5)


# In[65]:


df.drop(columns=['Memory'],inplace=True)


# In[67]:


df.head()


# In[68]:


df.corr()['Price']


# In[69]:


df.drop(columns=['Hybrid','Flash_Storage'],inplace=True)


# In[70]:


df.head()


# In[71]:


df['Gpu'].value_counts()


# In[72]:


df['Gpu brand'] = df['Gpu'].apply(lambda x:x.split()[0])


# In[73]:


df.head()


# In[74]:


df['Gpu brand'].value_counts()


# In[75]:


df = df[df['Gpu brand'] != 'ARM']


# In[76]:


df['Gpu brand'].value_counts()


# In[77]:


sns.barplot(x=df['Gpu brand'],y=df['Price'],estimator=np.median)
plt.xticks(rotation='vertical')
plt.show()


# In[85]:


df.drop(columns=['Gpu'],inplace=True)


# In[86]:


df.drop(columns=['Gpu'],inplace=True)
df.head()


# In[ ]:




