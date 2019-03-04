from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    
class scaler:
    
    def __init__(self, name='scaler'):
        self.name = name
    
    def feat_scaler(X_train, X_test, y_train, y_test, poly=False):
        ss = StandardScaler()
        X_train = ss.fit_transform(X_train)
        X_test = ss.transform(X_test)

        if poly == True:
            poly = PolynomialFeatures(degree=2) 
            X_train = poly.fit_transform(X_train)
            X_test = poly.fit_transform(X_test)

        return X_train, X_test, y_train, y_test