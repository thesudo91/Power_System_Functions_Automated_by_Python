import scipy as sp
import scipy.sparse  # call as sp.sparse
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#A = sp.sparse.eye(2, 2, 1)

# from google.colab import drive
# drive.mount('/content/drive')

# path1 = "/content/drive/MyDrive/SRP_Topology/Bus_Data_Matrix_Zeros_Ones.csv"
# df1 = pd.read_csv(path1)


path1 = "C:/Users/avarghe6/Dropbox (ASU)/PC (3)/Downloads/Bus_Data_Matrix_Zeros_Ones.csv"
df1 = pd.read_csv(path1)
Amat = np.matrix(df1)
#%%
Amatrix = Amat[0:165,0:165]

# print(Amatrix)
# print(f"Shape1: {Amatrix.shape}")

# A = np.asarray([[0,1,1],[1,0,1],[1,1,0]])
sA = sp.sparse.csr_matrix(Amatrix)  
G = nx.from_scipy_sparse_matrix(sA)

#A = sp.sparse.csr_matrix([[1, 1], [1, 2]])
G = nx.from_scipy_sparse_matrix(sA,create_using=nx.MultiGraph)
#G[1][1]

pos = nx.spring_layout(G, seed=200)
#nx.draw(G, pos, node_size = 70, font_size = 6,with_labels=True)
nx.draw_kamada_kawai(G, node_size = 10, font_size = 2,with_labels=True)
plt.savefig("graph.png", dpi=2000)
plt.show()