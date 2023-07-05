import numpy as np
import joblib

from sklearn.cluster import KMeans

class ColorDescriptorExtractor:
    def __init__(self, name="Color", kmeans=None, n_clusters=10):
        self.name = name
        self.n_clusters = n_clusters
        self.kmeans = None if kmeans is None else joblib.load(kmeans)
        self.color_lut = None
        self.histogram = None

    def concat_reshape_image_color(self, batch):
        train_data_color = []
        train_data_concat = []

        for img in batch:
            train_data_concat.append(img.reshape(1024, 3))

        train_data_concat = np.concatenate(train_data_concat)
        train_data_color = train_data_concat[np.random.choice(train_data_concat.shape[0], 5000, replace=False)]

        return train_data_color, train_data_concat

    def fit(self, batch):
        print(self.name + " fitting kmeans...")
        train_data_color, _ = self.concat_reshape_image_color(batch)

        if not self.kmeans:
            self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
            joblib.dump(self.kmeans, self.name + '_kmeans.joblib')
        
        self.kmeans.fit(train_data_color)
        self.color_lut = np.uint8(self.kmeans.cluster_centers_)
        print(self.name + " done fitting kmeans.")

    def compute_descriptor(self, batch):
        print(self.name + " computing descriptor...")
        if self.kmeans is None or self.color_lut is None:
            raise RuntimeError("The descriptor extractor is not fitted. Call the 'fit' method first.")

        color_histograms = []
        for img in batch:
            labels = self.kmeans.predict(img.reshape(1024, 3))
            color_histogram = np.bincount(labels, minlength=len(self.color_lut))
            color_histogram = color_histogram / np.sum(color_histogram)
            color_histograms.append(color_histogram)

        self.histogram = color_histograms
        print(self.name + " done computing descriptor.")
