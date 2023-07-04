import numpy as np
import joblib
import cv2

from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer

class BOVWDescriptorExtractor:
    def __init__(self, name="BOVW", kmeans=None, n_clusters=512):
        self.name = name
        self.kmeans = None if kmeans is None else joblib.load(kmeans)
        self.n_clusters = n_clusters
        self.image_descriptor = None
        self.data = None
        self.data_concat = None

    def compute_mean(self, batch):
        train_mean = batch.mean(axis=0)
        return batch - train_mean

    def compute_pca(self, batch):
        train_cov = np.dot(batch.T, batch)
        eigvals, eigvecs = np.linalg.eig(train_cov)
        perm = eigvals.argsort()
        pca_transform = eigvecs[:, perm[64:128]]
        return pca_transform
    
    def transform_data(self, batch):
        train_data_sift = []
        for img in tqdm(batch):
            sift = cv2.SIFT_create()
            _, des = sift.detectAndCompute(img, None)

            if des is None:
                des = np.zeros((16, 128))
            elif len(des) < 16:
                des = np.concatenate((des, np.zeros((16 - len(des), 128))))
            else:
                des = des[:16]

            train_data_sift.append(des)

        train_data_sift = np.array(train_data_sift)

        train_data_concat = np.concatenate(train_data_sift)
        train_data_concat = train_data_concat.astype(np.float32)

        train_mean = self.compute_mean(train_data_concat)


        self.data_concat = train_mean #pca_transform
        self.data = train_data_sift

    def fit(self, batch):
        print(self.name + " fitting kmeans...")
        if self.data is None:
            self.transform_data(batch)

        if self.kmeans == None:
            self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42).fit(self.data_concat)
            joblib.dump(self.kmeans, self.name + '_kmeans.joblib')   
        print(self.name + " fitting kmeans... done")      

    def compute_descriptor(self, batch):
        print(self.name + " computing descriptor...")
        if self.kmeans is None:
            raise RuntimeError("The descriptor extractor is not fitted. Call the 'fit' method first.")
        
        if self.data is None:
            self.transform_data(batch)

        l2_normalizer = Normalizer(norm='l2')

        image_descriptor = np.zeros((len(self.data), self.kmeans.n_clusters), dtype=np.float32)

        for index, desc in enumerate(self.data):
            desc = desc.astype(np.float32)

            clabels = self.kmeans.predict(desc)

            descr_hist = np.bincount(clabels, minlength=self.kmeans.n_clusters)

            descr_hist = descr_hist.astype(np.float32)

            descr_hist = np.sqrt(descr_hist)

            descr_hist = l2_normalizer.transform(descr_hist.reshape(1, -1))

            image_descriptor[index] = descr_hist

        self.image_descriptor = image_descriptor
        print(self.name + " computing descriptor... done")