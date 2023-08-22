# create a predefined nn model
import numpy as np

class Model_neural:
    
    def __init__(self, learning_rate=0.01, train_itter=1000):
        self.learning_rate = learning_rate
        self.train_itter = train_itter
        self.W = np.random.rand(1, 3) * 10
        self.b = np.random.rand(1) * 10
        self.comulative_errros = []
    
    def ReLu(self, Z):
        return np.maximum(0, Z)
    
    def deriv_ReLu(self, Z):
        return (Z > 0) * 1
    
    # return the prediction value
    # just need X and previous function
    # x = [
        # [x11 x12 x13],
        # [x21 x22 x23],
        # ....
    # ]
    def predict(self, X):
        Z1 = np.add(np.dot(self.W, X.T), self.b)
        a1 = self.ReLu(Z1)
        
        prediction = a1
        
        return prediction
    
    # return the gradient to repair the model
    # need the prediction and various function
    # x = [
        # [x11 x12 x13], x1
        # [x21 x22 x23], x2
        # ....
    # ]
    # y = [y1 y2 ...]
    def gradient_descent(self, x, y):
        a0 = x.T
        Z1 = np.dot(self.W, x.T) + self.b
        a1 = self.ReLu(Z1)
        
        Cost = np.square(a1 - y.T)
        dCost = 2 * (a1 - y.T)
        dW = dCost * self.deriv_ReLu(Z1) * a0
        db = (dCost * self.deriv_ReLu(Z1))
        
        return dW, db
    
    # just need the dW and db --> we can get from gradient descent function
    def update_param(self, dW, db):
        self.W -= dW * self.learning_rate
        self.b -= db * self.learning_rate
    
    # need the update param function and decide the size of batch
    # let say size of batch is the number of itter that will be the
    # limit to then we do the update for the parameters
    def Train(self, X, Y,  size_Y, batch=5):
        dW, db = 0, 0
        self.comulative_errros = []
        if size_Y >= batch:
            for itter in range(self.train_itter):
                rd_idx = np.random.randint(0, size_Y)
                x = X[rd_idx]
                y = Y[rd_idx]
                dW_, db_ = self.gradient_descent(x, y)
                dW += dW_
                db += db_
                # for every batch we are going to update the parameter
                if itter % batch == 0:
                    self.update_param(dW=dW, db=db)
                    pred_value = self.predict(X)
                    print(f"error(mse): {np.sum(np.square(pred_value - Y))}")
                    print(dW, db)
                    self.comulative_errros.append(np.sum(np.square(pred_value - Y)))
                    pass
        else:
            for itter in range(self.train_itter):
                rd_idx = np.random.randint(0, size_Y)
                x = X[rd_idx]
                y = Y[rd_idx]
                dW_, db_ = self.gradient_descent(x, y)
                dW += dW_
                db += db_
                self.update_param(dW=dW, db=db)
                pred_value = self.predict(X)
                print(f"error(mse): {np.sum(np.square(pred_value - Y))}")
                print(dW, db)
                self.comulative_errros.append(np.sum(np.square(pred_value - Y)))