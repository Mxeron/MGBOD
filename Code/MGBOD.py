import numpy as np
from scipy.io import loadmat
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
        FSet = 1 / (1 + cdist(centers[:,[j]], centers[:,[j]]))
        FSet_de = 1 / (1 + cdist(centers[:, b_de[:m - k]], centers[:,b_de[:m - k]]))
        FSet_as = 1 / (1 + cdist(centers[:, b_as[:m - k]], centers[:,b_as[:m - k]]))
        
        FSet[FSet < sigma] = 0
        FSet_de[FSet_de < sigma] = 0
        FSet_as[FSet_as < sigma] = 0
        for i in range(n_gb):
            weight[i, k] = np.sum(FSet[i, :]) / n_gb 
            FG_de[i, k] = np.sum(FSet_de[i, :]) / n_gb 
            FG_as[i, k] = np.sum(FSet_as[i, :]) / n_gb


    MFGAF = np.zeros(n_gb)
    for j in range(n_gb):
        MFGAF[j] = 1 - (np.cbrt(np.sum(weight[j, :]) / m)) * ((np.sum(FG_as[j, :] + FG_de[j, :])) / (2 * m))
        
    print("样本数 : {}, 生成粒球数 : {}".format(n, n_gb))
    OF = np.zeros(n)
    
    for idx, gb in enumerate(GBs):
        point_idxs = gb[:,-1].astype('int')
        OF[point_idxs] = MFGAF[idx]
    return OF


if __name__ == "__main__":
    load_data = loadmat(r"D:\outlier_datasets\Numerical\ODDS\cardio.mat")
    trandata = load_data['trandata']
    X = trandata[:,:-1]
    label = trandata[:,-1]
    ID = (X >= 1).all(axis=0) & (X.max(axis=0) != X.min(axis=0))
    scaler = MinMaxScaler()
    if any(ID):
        X[:, ID] = scaler.fit_transform(X[:, ID])
    best_score = 0
    out_factors = MGBOD(X, 0.8)
    print(out_factors)