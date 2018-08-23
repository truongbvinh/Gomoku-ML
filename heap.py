"""
Author: Vinh Truong
Implementation of a max heap
"""

class maxHeap:
    def __init__(self, iterable = None, key=None):
        """
        Implements max heap binary tree as a breadth first array
        This allows for offline implementation, and east insertion
        into the heap
        """
        self.key = key
        if iterable:
            self.tree = [x for x in iterable]
            for x in range(len(self.tree), -1, -1):
                self._percolate_down(x)
        else:
            self.tree=[]
    
    def add(self, value):
        """
        Add a value into the heap
        """
        self.tree.append(value)
        index = len(self.tree)-1
        while self._greater_than(self.tree[index], self.tree[index//2]):
            temp = self.tree[index]
            self.tree[index] = self.tree[index//2]
            self.tree[index//2] = temp
            index //= 2
    
    def pop(self):
        """
        Removes the value at the top of the heap, and reorganizes
        the heap
        """
        result = self.tree[0]
        self.tree[0] = self.tree[len(self.tree)-1]
        del self.tree[len(self.tree)-1]
        self._percolate_down(0)
        return result
    
    def _greater_than(self, value1, value2):
        """
        Returns True if value1 is greater than value2
        Accepts a key function
        """
        if self.key != None:
            return self.key(value1) > self.key(value2)
        else:
            return value1 > value2

    # def verify(self, index):
    #     if index*2 > len(self.tree)-2:
    #         return True
    #     if self.tree[index] < self.tree[index*2]:
    #         return False
        
    #     result = []
    #     result.append(self.verify(index*2))
    #     if index*2+1 < len(self.tree):
    #         if self.tree[index] < self.tree[index*2+1]:
    #             return False
    #         result.append(self.verify(index*2+1))
        
    #     return 
        
        
    
    def _switch_down(self, index):
        """
        Examines a node and its children, and switches if it needs to be switched
        returns the index of the switched value if it was switched
        else returns None
        """
        children = self._child_exists(index)
        compare = int()
        if children == 2:
            left = index*2+1
            compare = left if self._greater_than(self.tree[left], self.tree[left+1]) else left+1
        elif children == 1:
            compare = left
        else:
            return None
        if self._greater_than(self.tree[compare], self.tree[index]):
            temp = self.tree[index]
            self.tree[index] = self.tree[compare]
            self.tree[compare] = temp
            return compare
        return None
    
    def _child_exists(self, index):
        """
        Returns 2 if a right child exists (which implies a left child does too)
        Returns 1 if and only if a left child exists
        Returns 0 if there are no children
        """
        if index*2+2 < len(self.tree):
            return 2
        if index*2+1 < len(self.tree)-1:
            return 1
        return 0
    
    def _percolate_down(self, index):
        """
        Percolates a given index down the max heap
        """
        x = index
        while x != None:
            x = self._switch_down(x)
    

if __name__ == "__main__":
    x = maxHeap([6, 2, 3, 7, 1, 4, 9])
    print(x.tree)
    """
            6
        2       3
      7   1   4    9
    """
    x.add(10)
    print(x.tree)
    x.pop()
    print(x.tree)