import numpy as np
import matplotlib.pyplot as plt

def compute_descriptor_flat(batch, verbose=False):
    new_batch = []

    #count = 0
    rand_count = np.random.randint(0, len(batch))

    for img in batch:
        new_batch.append(img.flatten()) 
        
        '''
        if verbose and count == rand_count:
            plt.figure(figsize=(16,8))
            plt.subplot(1, 2, 1)
            plt.imshow(img.reshape(32, 32, 3))
            plt.subplot(1, 2, 2)
            plt.imshow(img.flatten().reshape(1024, 3))
            plt.show()
        count += 1
        '''
        
    train_data = np.array(new_batch)

    return train_data