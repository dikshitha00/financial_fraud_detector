import numpy as np
from sklearn.ensemble import IsolationForest

# create random training data
X = np.random.rand(1000, 2)

# train anomaly detection model
model = IsolationForest(contamination=0.05)
model.fit(X)

def check_fraud(amount, time):

    data = [[amount, time]]

    prediction = model.predict(data)

    if prediction[0] == -1:
        return True
    else:
        return False
