import itertools as itertools
from typing import List, Dict, Set, Tuple

class TransactionDataset:
    def __init__(self, transactions: List[Set[str]]):
        self.transactions = transactions
        
    def __len__(self):
        return len(self.transactions)
        
    def __getitem__(self, idx):
        return self.transactions[idx]

class Apriori:
    def __init__(self, dataset: TransactionDataset, min_support: float):
        self.dataset = dataset
        self.min_support = min_support
        self.frequent_itemsets = self._apriori()
        
    def _apriori(self) -> Dict[int, Set[frozenset]]:
        # Create a dictionary to store the frequent itemsets
        freq_itemsets = {}
        
        # Generate frequent itemsets of length 1
        itemsets = self._generate_itemsets(1)
        freq_itemsets[1] = set(itemset for itemset, support in itemsets.items() if support >= self.min_support)
        
        k = 2
        while len(freq_itemsets[k-1]) > 0:
            # Generate candidate itemsets of length k
            candidate_itemsets = self._generate_candidate_itemsets(freq_itemsets[k-1], k)
            
            # Count the support of each candidate itemset
            itemsets = self._count_itemsets(candidate_itemsets)
            
            # Add frequent itemsets of length k to the dictionary
            freq_itemsets[k] = set(itemset for itemset, support in itemsets.items() if support >= self.min_support)
            
            k += 1
            
        return freq_itemsets
    
    def _generate_itemsets(self, k: int) -> Dict[frozenset, float]:
        # Count the occurrence of each item
        item_counts = {}
        for transaction in self.dataset:
            for item in transaction:
                item_counts[item] = item_counts.get(item, 0) + 1
        
        # Generate itemsets of length k
        itemsets = {}
        for item, count in item_counts.items():
            itemset = frozenset([item])
            itemsets[itemset] = count / len(self.dataset)
        
        return itemsets
    
    def _generate_candidate_itemsets(self, itemsets: Set[frozenset], k: int) -> Set[frozenset]:
        candidate_itemsets = set()
        
        for itemset1 in itemsets:
            for itemset2 in itemsets:
                if len(itemset1.union(itemset2)) == k:
                    candidate_itemsets.add(itemset1.union(itemset2))
                    
        return candidate_itemsets
    
    def _count_itemsets(self, itemsets: Set[frozenset]) -> Dict[frozenset, float]:
        item_counts = {itemset: 0 for itemset in itemsets}
        
        for transaction in self.dataset:
            for itemset in itemsets:
                if itemset.issubset(transaction):
                    item_counts[itemset] += 1
                    
        itemsets_support = {itemset: count / len(self.dataset) for itemset, count in item_counts.items()}
        
        return itemsets_support
    
    def print_frequent_itemsets(self):
        for k, itemsets in self.frequent_itemsets.items():
            print(f"Frequent itemsets of size {k}:")
            for itemset in itemsets:
                support = sum(1 for t in self.dataset if itemset.issubset(t))
                print(f"{itemset} ({support}/{len(self.dataset)})")
    
class AssociationRules:
        def __init__(self, apriori: Apriori, min_confidence: float):
            self.apriori = apriori
            self.min_confidence = min_confidence
            self.rules = self._generate_rules()
            
        def _generate_rules(self) -> List[Tuple[frozenset, frozenset, float]]:
            rules = []
            
            for itemset_size in range(2, len(self.apriori.frequent_itemsets)):
                for itemset in self.apriori.frequent_itemsets[itemset_size]:
                    for subset in self._get_all_subsets(itemset):
                        if len(subset) > 0:
                            antecedent = subset
                            consequent = itemset.difference(subset)
                            
                            support_antecedent = self.apriori.frequent_itemsets[len(antecedent)][antecedent]
                            support_itemset = self.apriori.frequent_itemsets[len(itemset)][itemset]
                            confidence = support_itemset / support_antecedent
                            
                            if confidence >= self.min_confidence:
                                rules.append((antecedent, consequent, confidence))
                                
            return rules
        
        def _get_all_subsets(self, itemset: frozenset) -> List[frozenset]:
            subsets = []
            for size in range(1, len(itemset)):
                for subset in itertools.combinations(itemset, size):
                    subsets.append(frozenset(subset))
                    
            return subsets
        
        def print_association_rules(self):
            for rule in self.rules:
                antecedent, consequent, confidence = rule
                print(f"{antecedent} => {consequent} ({confidence:.2f})")