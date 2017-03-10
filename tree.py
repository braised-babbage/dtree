""" A simple implementation of a classification tree. """

from collections import namedtuple
from operator import le,eq
from math import isclose

# two kinds of tree nodes
Leaf = namedtuple('Leaf', ['data'])
Internal = namedtuple('Internal',
                      ['left','right', 'var', 'val'])


def proportion(data,var):
    c = sum([getattr(d,var) for d in data])
    return c / len(data)

def gini_index(data,var):
    """The gini index is the empirical variance of data.var"""
    p = proportion(data,var)
    return p*(1-p)

def compare_op(val):
    """Binary comparison for tree creation, based on val's type. """
    return eq if isinstance(val,bool) else le

def split(data,var,val):
    """ Split data (a list of objects) by comparing var to val. """
    left,right = [],[]
    op = compare_op(val)
    for d in data:
        lst = left if op(getattr(d,var),val) else right
        lst.append(d)
    return left,right

def split_quality(data,left,right,target):
    """ The quality of a split of data into left and right. """
    if left == [] or right == []:
        return 0
    n = len(data)
    pl,pr = len(left)/n, len(right)/n
    gd = gini_index(data,target)
    gl = gini_index(left,target)
    gr = gini_index(right,target)
    return gd - pl*gl - pr*gr
    
def isleft(node,x):
    """For an internal node, does x belong to the left child?"""
    op = compare_op(node.val)
    return op(getattr(x,node.var),node.val)

def best_split(data,predictors,target):
    """ Return the split (described by a variable and a value) that maximizes the improvement in gini index. """
    splits = [(best_split_on_var(data,var,target),var)
              for var in predictors]
    (q,val),var = max(splits)
    return var,val

def best_split_on_var(data,var,target):
    if isinstance(getattr(data[0],var),bool):
        split_fn = all_boolean_splits
    else:
        split_fn = all_numeric_splits
    splits = split_fn(data,var,target)
    return max(splits)

def all_numeric_splits(data,var,target):
    """ Numeric splitting breaks data into a left group, 
with var <= val, and a right group, with var > val. """
    splits = []
    for d in data:
        val = getattr(d,var)
        left,right = split(data,var,val)
        q = split_quality(data,left,right,target)
        if q == 0:
            val = None
        splits.append((q,val))
    return splits

def all_boolean_splits(data,var,target):
    """ Boolean splitting breaks data into a left group,
with var == True, and a right group, with var == False. """
    left,right = split(data,var,True)
    q = split_quality(data,left,right,target)
    if q == 0:
        return [(0,None)]
    return [(q,True)]

def build_tree(data,predictors,target):
    """Breadth first, recursive constructon of tree."""
    var,val = best_split(data,predictors,target)
    if val is None:
        return Leaf(data)
    left,right = split(data,var,val)
    return Internal(build_tree(left,predictors,target),
                    build_tree(right,predictors,target),
                    var,
                    val)
    
def predict(tree,x,target):
    """Predict x.target from tree."""
    if isinstance(tree,Leaf):
        return (proportion(tree.data,target) >= 0.5)
    branch = tree.left if isleft(tree,x) else tree.right
    return predict(branch,x,target)

    
class ClassificationTree:
    def __init__(self,data,predictors,target):
        """ A classification tree is built from three things:
              data       -- a list of python objects,
              predictors -- strings corresponding to attributes 
                            to be used for prediction,
              target     -- the attribute to predict. """
        self.data = data
        self.predictors = predictors
        self.target = target
        self.tree = build_tree(data,predictors,target)

    def predict(self,d):
        """ Predict d.target. """
        return predict(self.tree,d,self.target)
    
