# UNQ_C2
# GRADED FUNCTION: compute_gradient
def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray): Shape (m,) Input to the model (Population of cities) 
      y (ndarray): Shape (m,) Label (Actual profits for the cities)
      w, b (scalar): Parameters of the model  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    
    # Number of training examples
    m = x.shape[0]
    
    # You need to return the following variables correctly
    dj_dw = 0
    dj_db = 0
    
     
    for i in range(m):
        f_x = x[i]*w + b
        err = f_x - y[i] 
        dj_db = dj_db + err
        dj_dw = dj_dw + (err)*x[i]
     
    dj_db = (1 / (m))*dj_db
    dj_dw = (1 / ( m))*dj_dw
    
        
    return dj_dw, dj_db