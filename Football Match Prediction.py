#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


matches = pd.read_csv("C:\\Python Projects\\Football Winning Project\\football match dataset.csv", index_col = 0)


# In[3]:


# Displaying the first few rows of the dataset

matches.head()


# In[4]:


# Shape of the dataset
matches.shape


# In[5]:


#count of matches that each team has

matches["team"].value_counts()


# In[6]:


# Matches played by Liverpool

matches[matches["team"]=="Liverpool"]


# In[7]:


#no of matches there for each match week

matches["round"].value_counts()


# In[8]:


#datatypes of each column
matches.dtypes


# In[9]:


# Convert date column to datetime
matches["date"] = pd.to_datetime(matches["date"])


# In[10]:


matches


# In[11]:


matches.dtypes


# In[12]:


# Encoding 'venue' to numeric codes
matches["venue_code"] = matches["venue"].astype("category").cat.codes


# In[13]:


matches


# In[14]:


#Encoding 'opponent' to numeric codes
matches["opponent_code"] = matches["opponent"].astype("category").cat.codes


# In[15]:


matches


# In[16]:


# Extracting hour from time 
matches["hour"] = matches["time"].str.replace(":.+", "", regex = True).astype("int")
matches


# In[17]:


#encoding day of the week
matches["day_code"] = matches["date"].dt.dayofweek
matches


# In[18]:


# Creating binary target variable for win (1) or loss/draw (0)

matches["target"] = (matches["result"]  == "W").astype("int")
matches


# In[19]:


from sklearn.ensemble import RandomForestClassifier


# In[20]:


rf = RandomForestClassifier(n_estimators = 50, min_samples_split = 10, random_state = 1)


# In[21]:


# Splitting data into training sets

train = matches[matches["date"] < '2022-01-01'] 


# In[22]:


# Splitting data into testing sets

test = matches[matches["date"]>'2022-01-01']


# In[23]:


# Defining predictors and fitting the model

predictors =["venue_code","opponent_code","hour","day_code"]


# In[24]:


rf.fit(train[predictors], train["target"])


# In[25]:


# Making predictions and calculating accuracy

preds = rf.predict(test[predictors])


# In[26]:


from sklearn.metrics import accuracy_score


# In[27]:


acc = accuracy_score(test["target"], preds)


# In[28]:


acc


# In[29]:


# Confusion matrix and precision score
combined = pd.DataFrame(dict(actual = test["target"], prediction=preds))


# In[30]:


pd.crosstab(index = combined["actual"], columns = combined["prediction"])


# from sklearn.metrics import precision_score

# In[31]:


from sklearn.metrics import precision_score


# In[32]:


precision_score(test["target"], preds)


# In[33]:


#creating grouped matches to know how well the team played from the past few matches
grouped_matches = matches.groupby("team")


# In[34]:


group = grouped_matches.get_group("Manchester City")


# In[35]:


group


# In[36]:


def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats =  group[cols].rolling(3,closed ='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset = new_cols)
    return group


# In[37]:


cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]


# In[38]:


new_cols


# In[39]:


rolling_averages(group, cols, new_cols)


# In[40]:


matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))


# In[41]:


matches_rolling


# In[42]:


matches_rolling =  matches_rolling. droplevel('team')


# In[43]:


matches_rolling


# In[44]:


matches_rolling.index =  range(matches_rolling.shape[0])


# In[45]:


matches_rolling


# In[46]:


def make_prediction(data, predictors):
    train = data[data["date"] < '2022-01-01']
    test =  data[data["date"] > '2022-01-01']
    rf.fit(train[predictors],train["target"])
    preds = rf.predict(test[predictors])
    combined = pd. DataFrame(dict(actual= test["target"], predicted = preds), index = test.index)
    precision = precision_score(test["target"], preds)
    return combined, precision


# In[47]:


combined, precision = make_prediction(matches_rolling, predictors + new_cols)


# In[48]:


precision


# In[49]:


combined


# In[50]:


combined =  combined.merge(matches_rolling[["date","team","opponent","result"]], left_index = True, right_index = True)


# In[51]:


combined


# In[52]:


#combining home and away predictions


# In[53]:


#making team and opponent team names are in same matching. eg: team: wolverhampton but in opponent team it is like wolves
class MissingDict(dict):
    _missing_ = lambda self, key:key
map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Newcastle United" : "Newcastle Utd",
    "Tottenham Hotspur" : "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves"
    
}
mapping = MissingDict(**map_values)


# In[54]:


mapping["West Ham United"]


# In[55]:


combined["new_team"] = combined["team"]


# In[56]:


combined


# In[57]:


merged = combined.merge(combined, left_on = ["date", "new_team"], right_on=["date","opponent"])


# In[58]:


merged


# In[61]:


merged[(merged["predicted_x"]==1) & (merged["predicted_y"]==0)]["actual_x"].value_counts()


# In[62]:


15/23


# In[63]:


matches.columns


# In[ ]:




