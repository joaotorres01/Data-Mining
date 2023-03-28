from itertools import combinations
from collections import defaultdict

class Apriori:
    def __init__(self, min_support=0.5, min_confidence=0.5):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.itemsets = {}
        self.rules = []

    def fit(self, transactions):
        self.transactions = transactions
        self.itemsets[1] = self.get_frequent_items(transactions, 1)

        k = 2
        while self.itemsets[k - 1]:
            self.itemsets[k] = self.get_frequent_items(transactions, k)
            if not self.itemsets[k]:
                del self.itemsets[k]
                break
            k += 1

        for k in range(2, len(self.itemsets)):
            for itemset in self.itemsets[k]:
                self.generate_rules(itemset)

    def get_frequent_items(self, transactions, k):
        item_counts = defaultdict(int)
        for transaction in transactions:
            for itemset in combinations(transaction, k):
                item_counts[frozenset(itemset)] += 1

        frequent_itemsets = [itemset for itemset, count in item_counts.items() if count/len(transactions) >= self.min_support]

        return frequent_itemsets

    def generate_rules(self, itemset):
        if len(itemset) < 2:
            return

        for i in range(1, len(itemset)):
            for antecedent in combinations(itemset, i):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                confidence = self.get_confidence(antecedent, consequent)
                if confidence >= self.min_confidence:
                    self.rules.append((antecedent, consequent, confidence))

    def get_confidence(self, antecedent, consequent):
        antecedent_count = 0
        consequent_count = 0
        for transaction in self.transactions:
            if antecedent.issubset(transaction):
                antecedent_count += 1
                if consequent.issubset(transaction):
                    consequent_count += 1

        return consequent_count / antecedent_count