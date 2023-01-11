""" 
We exported the tree in Graphviz format using the export_graphviz exporter.
Prereq: graphviz (https://graphviz.org/download/)
The goal of a decision tree is to try to understand why a guy goes to the store, and apply that to new data later on.
The data and Output:
---------------------------------------------------------
day  | supplies  | weather   | worked?   | went shopping?
---------------------------------------------------------
1    | low       | sunny     | yes       | yes       
2    | high      | sunny     | yes       | no        
3    | med       | cloudy    | yes       | no        
4    | low       | raining   | yes       | no        
5    | low       | cloudy    | no        | yes       
6    | high      | sunny     | no        | no        
7    | high      | raining   | no        | no
8    | med       | cloudy    | yes       | no
9    | low       | raining   | yes       | no
10   | low       | raining   | no        | yes
11   | med       | sunny     | no        | yes
12   | high      | sunny     | yes       | no

Predicted Random Results:
---------------------------------------------------------
day  | supplies  | weather   | worked?   | went shopping?
---------------------------------------------------------
1    | low       | sunny     | no        | yes
2    | med       | sunny     | no        | yes
3    | low       | sunny     | yes       | yes
4    | high      | raining   | no        | no
5    | med       | raining   | yes       | no
  """

import graphviz
import itertools
import random
from sklearn.tree import DecisionTreeClassifier, export_graphviz 
from sklearn.preprocessing import OneHotEncoder

# The possible value for each class
classes = {
    'supplies':['low','med','high'],
    'weather':['raining','cloudy','sunny'],
    'worked?':['yes','no']
}

# Our example data
data = [
    ['low',  'sunny',   'yes'],
    ['high', 'sunny',   'yes'],
    ['med',  'cloudy',  'yes'],
    ['low',  'raining', 'yes'],
    ['low',  'cloudy',  'no' ],
    ['high', 'sunny',   'no' ],
    ['high', 'raining', 'no' ],
    ['med',  'cloudy',  'yes'],
    ['low',  'raining', 'yes'],
    ['low',  'raining', 'no' ],
    ['med',  'sunny',   'no' ],
    ['high', 'sunny',   'yes']
]

# Our target variable, whether someone went shopping, Corelate with data
target = ['yes', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'no', 'yes', 'yes', 'no']

# Scikit learn can't handle categorical data, so form numeric representations of the above data

categories = [classes['supplies'], classes['weather'], classes['worked?']]
encoder = OneHotEncoder(categories=categories)

x_data = encoder.fit_transform(data)

# Form and fit our decision tree to the now-encoded data
classifier = DecisionTreeClassifier()
tree = classifier.fit(x_data, target)

# Now that we have our decision tree, let's predict some outcomes from random data
# This goes through each class and builds a random set of 5 data points
prediction_data = []
for _ in itertools.repeat(None, 5):
    prediction_data.append([
        random.choice(classes['supplies']),
        random.choice(classes['weather']),
        random.choice(classes['worked?'])
    ])

# Use our tree to predict the outcome of the random values
prediction_results = tree.predict(encoder.transform(prediction_data))



# =============================================================================
# Output code

def format_array(arr):
    return "".join(["| {:<10}".format(item) for item in arr])

def print_table(data, results):
    line = "day  " + format_array(list(classes.keys()) + ["went shopping?"])
    print("-" * len(line))
    print(line)
    print("-" * len(line))

    for day, row in enumerate(data):
        print("{:<5}".format(day + 1) + format_array(row + [results[day]]))
    print("")

feature_names = (
    ['supplies-' + x for x in classes["supplies"]] +
    ['weather-' + x for x in classes["weather"]] +
    ['worked-' + x for x in classes["worked?"]]
)

# Shows a visualization of the decision tree using graphviz
# Note that sklearn is unable to generate non-binary trees, so these are based on individual options in each class
dot_data = export_graphviz(tree, filled=True, proportion=True, feature_names=feature_names) 
graph = graphviz.Source(dot_data)
graph.render(filename='supervised_learning_DecisionTree', cleanup=True, view=True)

# Display out training and prediction data and results
print("Training Data:")
print_table(data, target)

print("Predicted Random Results:")
print_table(prediction_data, prediction_results)