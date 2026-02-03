import numpy as np

class rLDA:
    """
    Binary rLDA with covariance shrinkage:
        S_reg = (1-lam)*S + lam*I
    """
    def __init__(self, lam=0.1):
        self.lam = lam
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        X: (N, D)
        y: (N,)
        """
        X0 = X[y == 0]
        X1 = X[y == 1]

        mu0 = X0.mean(axis=0)
        mu1 = X1.mean(axis=0)

        # pooled covariance (bias=True = divide by N)
        S0 = np.cov(X0, rowvar=False, bias=True)
        S1 = np.cov(X1, rowvar=False, bias=True)
        S = 0.5 * (S0 + S1)
        
        D = S.shape[0]
        S_reg = (1.0 - self.lam) * S + self.lam * np.eye(D)

        invS = np.linalg.pinv(S_reg)
        self.w = invS @ (mu1 - mu0)

        # Equal priors => bias term
        self.b = -0.5 * (mu0 + mu1) @ self.w
        return self

    def predict(self, X):
        scores = X @ self.w + self.b
        return (scores > 0).astype(int)
    
if __name__ == "__main__":
    """
    LDA = Linear Discriminant Analysis
    
    The d-dimensional feature vector x is projected onto a lower dimension (scalar here) such that the
    projected means/averages of the classes are far apart, while the spread of the projected data is small.
    
    This can be realized by optimizing a cost function related to within-class
    covariance matrix (SW) and between-class covariance matrix (SB)
    
    The mapping maximizes the criterion function:
        J(w) = (w^T SB w) / (w^T SW w)
        
        
    rLDA = Regularized Linear Discriminant Analysis
    
    The difference between rLDA and LDA is that the covariance matrix is regularized
    by adding a multiple of the identity matrix to it. This helps to improve the
    stability and generalization of the model, especially when dealing with high-dimensional
    data or when the number of samples is limited.
    
    
    Steps:
    1. estimate class means
    2. estimate pooled covariance
    3. Regularize covariance with shrinkage to identity
    4. compute LDA weight vector w
    5. classify by thresholding the linear score
    
    """
    
    pass