#!/usr/bin/env python
# coding: utf-8

# # **Custom Transformer**

# In[7]:


from custom_transformers import CombinedAttributesAdder


# In[4]:


# import numpy as np
# from sklearn.base import BaseEstimator, TransformerMixin

# rooms_ix, bedrooms_ix, pop_ix, households_ix = 3, 4, 5, 6

# class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
#     def __init__(self, add_bedrooms_per_room=True):
#         self.add_bedrooms_per_room = add_bedrooms_per_room

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
#         pop_per_household = X[:, pop_ix] / X[:, households_ix]

#         if self.add_bedrooms_per_room:
#             bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
#             return np.c_[X, rooms_per_household, pop_per_household, bedrooms_per_room]
#         else:
#             return np.c_[X, rooms_per_household, pop_per_household]


# In[ ]:




