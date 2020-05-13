import numpy as np

def generateAdjacenyMatrix(n, p):
    '''
    Helper function to generate a sample adjaceny matrix where the prob of person i,j connecting is p
    '''
    
    #check for valid p
    if p < 0 or p > 1:
        raise Exception('Invalid p')
    
    #Adjusts p to deal with the fact that we're generating a symmetric matrix by doing mat + mat.T
    p_new = 1 - np.sqrt(1-p)
    
    #Compute random matrix with p' and zero out diagonal
    mat = np.random.binomial(1, p = p_new, size = (n,n))
    np.fill_diagonal(mat, 0)
    
    #Return symmetrized matrix
    return np.ceil((mat + mat.T)/2)