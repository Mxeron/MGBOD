import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from GB_generation_with_idx import get_GB


def MGBOD(data, sigma):
    n, m = data.shape
    GBs = get_GB(data)
    n_gb = len(GBs)
    centers = np.zeros((n_gb, m))
    for idx, gb in enumerate(GBs):
        centers[idx] = np.mean(gb[:,:-1], axis=0)
    E = np.zeros(m)
    
    for j in range(m):
        r1 = 1 / (1 + cdist(centers[:,[j]], centers[:,[j]]))
        r1[r1 < sigma] = 0
        E[j] = -(1 / n_gb) * np.sum(np.log2(np.sum(r1, axis=1) / n_gb))
    b_de = np.argsort(E)[::-1]
    b_as = np.argsort(E)

    weight = np.zeros((n_gb, m))
    FG_de = np.zeros((n_gb, m))
    FG_as = np.zeros((n_gb, m))

    for k in range(m):
        FSet = 1 / (1 + cdist(centers[:,[k]], centers[:,[k]]))
        FSet_de = 1 / (1 + cdist(centers[:, b_de[:m - k]], centers[:,b_de[:m - k]]))
        FSet_as = 1 / (1 + cdist(centers[:, b_as[:m - k]], centers[:,b_as[:m - k]]))
        
        FSet[FSet < sigma] = 0
        FSet_de[FSet_de < sigma] = 0
        FSet_as[FSet_as < sigma] = 0
        for i in range(n_gb):
            weight[i, k] = np.sum(FSet[i, :]) / n_gb 
            FG_de[i, k] = np.sum(FSet_de[i, :]) / n_gb 
            FG_as[i, k] = np.sum(FSet_as[i, :]) / n_gb

    OF_gb = np.zeros(n_gb)
    for j in range(n_gb):
        OF_gb[j] = 1 - (np.cbrt(np.sum(weight[j, :]) / m)) * ((np.sum(FG_as[j, :] + FG_de[j, :])) / (2 * m))
    OF = np.zeros(n)
    for idx, gb in enumerate(GBs):
        point_idxs = gb[:,-1].astype('int')
        OF[point_idxs] = OF_gb[idx]
    return OF

if __name__ == '__main__':
    data = pd.read_csv("./Example.csv").values
    ID = (data >= 1).all(axis=0) & (data.max(axis=0) != data.min(axis=0))
    scaler = MinMaxScaler()
    if any(ID):
        data[:, ID] = scaler.fit_transform(data[:, ID])
    sigma = 0.6
    out_factors = MGBOD(data, sigma)
    print(out_factors)