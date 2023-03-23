import pandas as pd

#1: Load the tennis dataset and separate the attributes (features) and the labels (target variable).
tennis = pd.read_csv('tennis.csv')
X = tennis.iloc[:, :-1] # attributes
y = tennis.iloc[:, -1]  # labels

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

#3: Test the PRIM algorithm for a specific class (e.g., "Yes" class) and print the induced rule.
target_class = "Yes"
rule = prim(X, y, target_class)
print(f"Rule for {target_class} class: {rule}")

#4: Repeat Step 3 for each class in the target variable.
for target_class in y.unique():
    rule = prim(X, y, target_class)
    print(f"Rule for {target_class} class: {rule}")
