import numpy as np

'''
#1: Load the tennis dataset and separate the attributes (features) and the labels (target variable).
tennis = pd.read_csv('tennis.csv')
X = tennis.iloc[:, :-1] # attributes
y = tennis.iloc[:, -1]  # labels
'''

class Prism:
    def __init__(self, target_class):
        self.target_class = target_class
        self.selected_attributes = []
    
    def fit(self, X, y):
        while len(X) > 0:
            # Step 1: Calculate the probability of each attribute/value pair
            prob = np.empty((X.shape[1], np.unique(X).size), dtype=float)
            prob[:] = np.nan
            for i, col in enumerate(range(X.shape[1])):
                for j, val in enumerate(np.unique(X[:,col])):
                    prob[i,j] = np.sum((X[:,col] == val) & (y == self.target_class)) / np.sum(y == self.target_class)

            # Step 2: Select the pair with the largest probability
            attribute, value = np.unravel_index(np.nanargmax(prob), prob.shape)
            self.selected_attributes.append((attribute, value))

            # Step 3: Create a subset of the training set with the selected attribute/value combination
            mask = X[:, attribute] == value
            X = X[mask, :]
            y = y[mask]

            # Step 4: Remove all instances covered by this rule from the training set
            mask = X[:, attribute] != value
            X = X[mask, :]
            y = y[mask]
    
    def get_rule(self):
        # Construct the induced rule as the conjunction of all selected attribute/value pairs
        rule = " and ".join([f"{att} == '{val}'" for att, val in self.selected_attributes])
        return rule

'''
#2: Implement the PRIM algorithm for a specific class (e.g., "Yes" class). Repeat this process for each class in the target variable.
def prim(X, y, target_class):
    selected_attributes = []
    while len(X) > 0:
        # Step 1: Calculate the probability of each attribute/value pair
        prob = pd.DataFrame(index=X.columns, columns=X.iloc[0,:])
        for col in X.columns:
            for val in X[col].unique():
                prob.loc[col,val] = len(X[(X[col] == val) & (y == target_class)]) / len(X[y == target_class])
        
        # Step 2: Select the pair with the largest probability
        attribute, value = prob.stack().idxmax()
        selected_attributes.append((attribute, value))
        
        # Step 3: Create a subset of the training set with the selected attribute/value combination
        X = X[X[attribute] == value]
        y = y[X.index]
        
        # Step 4: Remove all instances covered by this rule from the training set
        X = X[X[attribute] != value]
        y = y[X.index]
    
    # Construct the induced rule as the conjunction of all selected attribute/value pairs
    rule = " and ".join([f"{att} == '{val}'" for att, val in selected_attributes])
    return rule
    '''
'''
#3: Test the PRIM algorithm for a specific class (e.g., "Yes" class) and print the induced rule.
target_class = "Yes"
rule = prim(X, y, target_class)
print(f"Rule for {target_class} class: {rule}")
'''
'''
#4: Repeat 3 for each class in the target variable.
for target_class in y.unique():
    p = Prism(target_class=target_class)
    p.fit(X, y)
    rule = p.get_rule()
    print(f"Rule for {target_class} class: {rule}")
'''