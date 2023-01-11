import graphviz
import itertools
import random
from sklearn.tree import DecisionTreeClassifier, export_graphvizfrom sklearn.preprocessing import OneHotEncoder

# The possible value for each class
classes = {
    'supplies':['low','med','high'],
    'weather':['raining','cloudy','sunny'],
    'worked?':['yes','no']
}