from tree import *
from math import isclose
from collections import namedtuple

S = namedtuple('S', ['i','b'])
data1 = [S(1,True), S(2,False), S(3,True)]
data2 = [S(i,True) for i in [1,2]]
data3 = [S(3,False)]

def test_prop():
    assert(isclose(proportion(data1,'b'),2/3))
    assert(isclose(proportion(data2,'b'),1))
    assert(isclose(proportion(data3,'b'),0))

def test_gini():
    assert(isclose(gini_index(data1,'b'),2/9))
    assert(isclose(gini_index(data2,'b'),0))
    assert(isclose(gini_index(data3,'b'),0))

D = namedtuple('D', ['i','b','y'])
data4 = [D(i,True,True) for i in [1,2]]
data5 = [D(3,False,False)]
data6 = data4 + data5
tree1 = Leaf(data4)
tree2 = Leaf(data5)
tree3 = Internal(tree1,tree2,var='b',val=True)

def test_isleft():
    assert(getattr(D(0,False,False),'b') == False)
    assert(tree3.val)
    assert(isleft(tree3,D(0,True,None)))
    assert(isleft(tree3,D(0,False,None)) == False)

def test_predict():
    d = D(0,False,None)
    assert(predict(tree1,d,'y') == True)
    assert(predict(tree2,d,'y') == False)
    assert(predict(tree3,d,'y') == False)


def test_split():
    assert(split(data6,'i',2) == (data4,data5))
    assert(split(data6,'b',True) == (data4,data5))

def test_quality():
    assert(isclose(split_quality(data6,data4,data5,'y'),
                   gini_index(data6,'y')))
    assert(isclose(split_quality(data6,[],data6,'y'),0))
    
    data = [D(0,True,False), D(1,True,False),
            D(2,True,False), D(3,False,True)]
    assert(isclose(gini_index(data,'y'),0.1875))
    left,right = data[:1],data[1:]
    assert(isclose(split_quality(data,left,right,'y'),
                   0.1875 - (3/4)*gini_index(right,'y')))
    

def test_splits():
    data = [S(0,True),S(0,False),S(0,True)]
    assert(len(set(all_numeric_splits(data,'i','b'))) == 1)
    assert(len(set(all_boolean_splits(data,'b','b'))) == 1)

    data = [S(0,True),S(0,False),S(1,True)]
    assert(len(set(all_numeric_splits(data,'i','b'))) == 2)


def test_best_split():
    data = [D(i=0, b=True, y=False), D(i=1, b=True, y=False),
            D(i=2, b=True, y=False), D(i=3, b=False, y=True)]
    assert(best_split(data,['i','b'],'y') == ('i',2))
