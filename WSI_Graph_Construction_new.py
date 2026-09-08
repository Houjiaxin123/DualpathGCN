### System
import os
import sys
import h5py
from itertools import chain
from tqdm import tqdm

### Graph Network Packages
import nmslib

### PyTorch / PyG
from torch_geometric.data import Data as geomData

### CLAM Path
clam_path = '/root/Desktop/data/private/CLAM0516'
sys.path.append(clam_path)
from utils.utils import *


class Hnsw:
    def __init__(self, space='cosinesimil', index_params=None,
                 query_params=None, print_progress=True):
        self.space = space
        self.index_params = index_params
        self.query_params = query_params
        self.print_progress = print_progress

    def fit(self, X):
        index_params = self.index_params
        if index_params is None:
            index_params = {'M': 16, 'post': 0, 'efConstruction': 400}

        query_params = self.query_params
        if query_params is None:
            query_params = {'ef': 90}

        index = nmslib.init(space=self.space, method='hnsw')
        index.addDataPointBatch(X)
        index.createIndex(index_params, print_progress=self.print_progress)
        index.setQueryTimeParams(query_params)

        self.index_ = index
        self.index_params_ = index_params
        self.query_params_ = query_params
        return self

    def query(self, vector, topn):
        indices, dist = self.index_.knnQuery(vector, k=topn)
        return indices


def build_knn_edge_index(X, radius=9, space='l2', print_progress=False):
    assert X.ndim == 2, f"Expected X to be 2D, got shape {X.shape}"

    num_nodes = X.shape[0]
    model = Hnsw(space=space, print_progress=print_progress)
    model.fit(X)

    src = np.repeat(np.arange(num_nodes), radius - 1)
    dst = np.fromiter(
        chain(*[model.query(X[i], topn=radius)[1:] for i in range(num_nodes)]),
        dtype=np.int64
    )

    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    return edge_index


def pt2graph(wsi_h5, radius=9, spatial_space='l2', latent_space='cosinesimil', print_progress=False):
    coords = np.array(wsi_h5['coords'])  # [N, 2]
    features = np.array(wsi_h5['features'])  # [N, 1024]

    assert coords.shape[0] == features.shape[0], \
        f"coords and features must have same number of patches, got {coords.shape[0]} and {features.shape[0]}"
    assert coords.ndim == 2 and coords.shape[1] == 2, \
        f"coords should have shape [N, 2], got {coords.shape}"
    assert features.ndim == 2, \
        f"features should have shape [N, d], got {features.shape}"

    # spatial graph:
    edge_spatial = build_knn_edge_index(
        X=coords,
        radius=radius,
        space=spatial_space,
        print_progress=print_progress
    )

    # latent graph:
    edge_latent = build_knn_edge_index(
        X=features,
        radius=radius,
        space=latent_space,
        print_progress=print_progress
    )

    G = geomData(
        x=torch.tensor(features, dtype=torch.float),
        edge_index=edge_spatial,  # spatial graph
        edge_latent=edge_latent,  # latent / feature graph
        centroid=torch.tensor(coords, dtype=torch.float)
    )

    return G


def createDir_h5toPyG(h5_path, save_path, radius=9, spatial_space='l2', latent_space='cosinesimil'):
    os.makedirs(save_path, exist_ok=True)

    pbar = tqdm(os.listdir(h5_path))
    for h5_fname in pbar:
        pbar.set_description(f'{h5_fname[:12]} - Creating Graph')

        try:
            h5_file_path = os.path.join(h5_path, h5_fname)
            with h5py.File(h5_file_path, "r") as wsi_h5:
                G = pt2graph(
                    wsi_h5,
                    radius=radius,
                    spatial_space=spatial_space,
                    latent_space=latent_space,
                    print_progress=False
                )
                torch.save(G, os.path.join(save_path, h5_fname[:-3] + '.pt'))

        except OSError:
            pbar.set_description(f'{h5_fname[:12]} - Broken H5')
            print(h5_fname, 'Broken')


h5_path = '/root/Desktop/data/private/LIHC/feature_DX_UNI2_RE/h5_files'
save_path = '/root/Desktop/data/private/LIHC/patch_graph_343_uni2'

createDir_h5toPyG(
    h5_path=h5_path,
    save_path=save_path,
    radius=9,
    spatial_space='l2',
    latent_space='cosinesimil'
)