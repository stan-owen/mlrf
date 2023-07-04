import numpy as np
import matplotlib.pyplot as plt


class FlatDescriptorExtractor:
    def __init__(self, name="Flat"):
        self.name = name
        self.train_data = None

    def compute_descriptor_flat(self, batch):
        print(self.name + " computing descriptor...")
        new_batch = []

        #count = 0
        rand_count = np.random.randint(0, len(batch))

        for img in batch:
            new_batch.append(img.flatten()) 
            
        train_data = np.array(new_batch)

        self.train_data = train_data
        print(self.name + " done computing descriptor.")