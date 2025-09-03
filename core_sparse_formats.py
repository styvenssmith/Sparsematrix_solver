import numpy as np

class CSRMatrix:
    def __init__(self, values, col_indices, row_ptr, shape):
        
        self.data = np.array(values)
        self.indices = np.array(col_indices)
        self.index_ptr = np.array(row_ptr)
        self.shape = shape
        

  #allows us to construct a CSR object without using the __init__ constructor
    @classmethod
    def from_dense(cls, data):
      
      rows, cols = len(data), len(data[0])
      vals = []
      col_indices = []
      row_ptr = [0]

      for i in range(rows):
        temp = 0
        for j in range(cols):
          if(data[i][j]!=0):
            vals.append(data[i][j])
            col_indices.append(j)
            temp = temp+1
        row_ptr.append(temp+row_ptr[-1])

      return cls(vals, col_indices, row_ptr, (rows, cols))


    # allows us to multiply the matrix by vectors
    def dot(self, vec):

        #check that the columns of the matrix matches the rows of the vector

      if len(vec)!=self.shape[1]:
        raise ValueError("The dimensions do not match")

        #since the vector will be nx1
        #the matrix is mxn, therefore the result will be mx1, we only need the rows of the matrix
      ans = np.zeros(self.shape[0])

      for i in range(self.shape[0]):
        start = self.index_ptr[i]
        end = self.index_ptr[i+1]
        for j in range(start, end):
          ans[i]+=self.data[j]*vec[self.indices[j]]

      return ans





#test. [1, 0], [0, 2]

data = [1, 2]
indices = [0, 1]
index_ptr = [0,  1, 2]
shape = (2, 2)

matrix = [
    [5, 0, 0, 1],  # row 0
    [0, 8, 0, 0],  # row 1  
    [0, 0, 3, 0]   # row 2
]

obj2 = CSRMatrix.from_dense(matrix)

obj2.dot([1,2,3,4])