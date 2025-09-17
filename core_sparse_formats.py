import numpy as np

class CSCMatrix:
  def __init__(self,values,row_indices, col_ptr, shape):
    self.data = np.array(values)
    self.indices = np.array(row_indices)
    self.col_ptr = np.array(col_ptr)
    self.shape = shape

    def __getitem__(self, key):
      row, col = key
    # Check if row is within bounds
      if row < 0 or row >= self.shape[0]:
          raise IndexError(f"Row index {row} out of range")
      if col < 0 or col >= self.shape[1]:
          raise IndexError(f"Column index {col} out of range")

    # Get the start and end indices for the given row
      start = self.index_ptr[row]
      end = self.index_ptr[row+1]

    # Iterate over the non-zero elements in this row
      for idx in range(start, end):
          if self.indices[idx] == col:  # Check if column matches
              return self.data[idx]     # Return the value

    # If no non-zero element found at (row, col), return 0
      return 0
     
     
    def to_dense(self):
      dense = np.zeros(self.shape)
      for j in range(self.shape[1]):
        start, end =  self.col_ptr[i], self.col_ptr[i+1]
        for idx in range(start, end):
          i= self.row_indices[idx]
          dense[i, j] = self.values[idx]
      return dense
   


class CSRMatrix:
    def __init__(self, values, col_indices, row_ptr, shape):
        self.data = np.array(values)
        self.indices = np.array(col_indices)
        self.index_ptr = np.array(row_ptr)
        self.shape = shape

    def __getitem__(self, key):
      row, col = key
    # Check if row is within bounds
      if row < 0 or row >= self.shape[0]:
          raise IndexError(f"Row index {row} out of range")
      if col < 0 or col >= self.shape[1]:
          raise IndexError(f"Column index {col} out of range")

    # Get the start and end indices for the given row
      start = self.index_ptr[row]
      end = self.index_ptr[row+1]

    # Iterate over the non-zero elements in this row
      for idx in range(start, end):
          if self.indices[idx] == col:  # Check if column matches
              return self.data[idx]     # Return the value

    # If no non-zero element found at (row, col), return 0
      return 0

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

  #need to work on this
    def to_csc(self):
       if not hasattr(self, 'shape') or not hasattr(self, 'data') or not hasattr(self, 'indices')or not hasattr(self, 'index_ptr'):
          raise ValueError('Matrix is not in proper CSR format')
       
      

       
    

    #convert to dense matrix
    def to_dense(self):

      dense = np.zeros(self.shape)
      for i in range(self.shape[0]):
        start=self.index_ptr[i]
        end = self.index_ptr[i+1]
        for j in range(start, end):
          dense[i, self.indices[j]] =self.data[j]
      return dense


    #perform matrix by matrix multiplications
    def matmat(self, matrix):
      
      #input validation
      if not hasattr(matrix, 'shape') and not hasattr(matrix, '__len__'):
         raise TypeError("Input must be a matrix or 2d array")
      
      if hasattr(matrix, 'shapses'):
         if len(matrix.shape)!=2:
            raise ValueError("input matrix must be 2-dimensional")
         if self.shape[1]!=matrix.shape[0]:
            raise ValueError(f"matrix dimensions incompatible: {self.shape[1]}!={matrix.shape[0]}")
         
         matrix_array = np.asarray(matrix)
         n_cols = matrix_array.shape[1]

      else:
         if not all(hasattr(row, '__len__') for row in matrix):
            raise ValueError("input must be a 2d matrix structure")
         
         n_rows = len(matrix)
         if n_rows==0:
            raise ValueError('input  matrix cannot be empty')
         n_cols  = len(matrix[0])
         if not all(len(row)==n_cols for row in matrix):
            raise ValueError("all rows in input matrix must have the same length")
         if self.shape[1]!=n_rows:
            raise ValueError(f'matrix dimensionss incompatible: {self.shape[1]}!=n_rows')
         
         matrix_array=np.array(matrix)
      
      result = np.zeros((self.shape[0], n_cols))


      #if the col len of our csr matrix is not matching the rows of the new matrix, cannot be done
     
      for i in range(self.shape[0]):
        start = self.index_ptr[i]
        end = self.index_ptr[i+1]
        for idx in range(start, end):
          j=self.indices[idx]
          value =self.data[idx]

          for k in range(len(matrix[0])):
            result[i,  k]+=value*matrix[j][k]

      return result








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

dot_product= obj2.dot([1,2,3,4])

obj= CSRMatrix(data, indices,index_ptr, shape)

n = [[1,0,0],
     [2,0,0],
     [3,0,0],
     [4,0,0]]
obj2.matmat(n)