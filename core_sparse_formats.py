class SparseMatrix:
    def __init__(self, shape: tuple[int, int], dtype=float):
        self.shape = shape
        self.rows, self.cols = shape
        self.dtype = dtype
        self.format = None


#implement coo class and functions

class COOMatrix(SparseMatrix):
    def __init__(self, dense_matrix):
        self.shape = (len(dense_matrix), len(dense_matrix[0]))
        self.data= []
        self.row = []
        self.col = []
        self.nnz = len(self.data)

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if(dense_matrix[i][j]!=0):
                    self.data.append(dense_matrix[i][j])
                    self.row.append(i)
                    self.col.append(j)



    #conversion methods
    def tocsr(self):
        return

    def tocsc(self):
        return
    
    def tocoo(self):
        return
    
    def __add__(self, other):
        return
    


 





class CSRMatrix(SparseMatrix):
    def __init__(self, shape: tuple[int,int] = (0,0), dense_matrix=None):
        if dense_matrix is not None:
            # Extract shape and dtype from dense_matrix
            shape = (len(dense_matrix), len(dense_matrix[0]))
            dtype = type(dense_matrix[0][0])
            super().__init__(shape, dtype)   # Initialize parent
        else:
            super().__init__(shape)           # Initialize parent with given shape

        self.values = []
        self.row_indices = []
        self.row_ptr = [0]
        self.format = 'CSR'
      

    
    def _from_dense(self, dense_matrix):
        """Convert a dense matrix to CSR format"""
    
        for i in range(self.shape[0]):
            row_nonzero = 0
            for j in range(self.cols):
                if dense_matrix[i][j] != 0:
                    self.values.append(dense_matrix[i][j])
                    self.col_indices.append(j)
                    row_nonzero += 1
            self.row_ptr.append(self.row_ptr[-1] + row_nonzero)
    
    def get_row(self, row_idx):
        """Get a specific row as a dense array"""
        if row_idx < 0 or row_idx >= self.shape[0]:
            raise IndexError("Row index out of bounds")
            
        start = self.row_ptr[row_idx]
        end = self.row_ptr[row_idx + 1]
        row = [0] * self.cols
        
        for i in range(start, end):
            col = self.col_indices[i]
            row[col] = self.values[i]
            
        return row

    def to_dense(self):
        """Convert CSR matrix to dense format"""
        dense = [[0] * self.cols for _ in range(self.shape[0])]
        
        for row_idx in range(self.shape[0]):
            start = self.row_ptr[row_idx]
            end = self.row_ptr[row_idx + 1]
            
            for i in range(start, end):
                col_idx = self.col_indices[i]
                dense[row_idx][col_idx] = self.values[i]
                
        return dense
    
    def matvec(self, vector):
        """Matrix-vector multiplication"""
        if len(vector) != self.cols:
            raise ValueError("Vector dimension doesn't match matrix columns")
            
        result = [0] * self.shape[0]
        
        for i in range(self.shape[0]):
            start = self.row_ptr[i]
            end = self.row_ptr[i + 1]
            
            for j in range(start, end):
                result[i] += self.values[j] * vector[self.col_indices[j]]
                
        return result
    
    def __str__(self):
        return f"CSRMatrix(shape={self.shape}, nnz={len(self.values)})"

# Example usage
if __name__ == "__main__":
    # Create a dense matrix
    dense_matrix = [
        [1, 0, 0, 2],
        [0, 0, 3, 0],
        [0, 4, 0, 5],
        [6, 0, 0, 7]
    ]
    

    csc_matrix= CSCMatrix(dense_matrix= dense_matrix)
    
    # Convert to CSR format
    csr_matrix = CSRMatrix(dense_matrix=dense_matrix)
    print(csr_matrix)
    
    # Convert back to dense
    reconstructed = csr_matrix.to_dense()
    print("Reconstructed dense matrix:")
    for row in reconstructed:
        print(row)
    
    # Test matrix-vector multiplication
    vector = [1, 2, 3, 4]
    result = csr_matrix.matvec(vector)
    print(f"Matrix-vector product: {result}")
    
    # Get a specific row
    row_2 = csr_matrix.get_row(2)
    print(f"Row 2: {row_2}")
    
    # Display CSR internal structure
    print(f"\nCSR internal structure:")
    print(f"Values: {csr_matrix.values}")
    print(f"Column indices: {csr_matrix.col_indices}")
    print(f"Row pointers: {csr_matrix.row_ptr}")