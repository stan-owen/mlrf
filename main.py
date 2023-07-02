from color import *
from bovw import *
from data import *
from result import *

from sklearn.ensemble import RandomForestClassifier

def __main__():
    print("Loading data...")
    train_batch, train_labels, test_batch, labels_test, labels_name = load_data()
    print(train_batch.shape)
    print("Data loaded")

    print("Computing descriptor BOVW...")
    Bovw = BOVWDescriptorExtractor(joblib.load(f'BOVW_kmeans.joblib'))
    print("Fitting...")
    Bovw.fit(train_batch)
    print("Computing...")
    Bovw.compute_descriptor(train_batch)
    print("Descriptor computed")

    rfc = RandomForestClassifier(n_estimators=500, max_depth=50, random_state=42)
    rfc.fit(Bovw.image_descriptor, train_labels)

    plot_results("BOVW", rfc, Bovw.image_descriptor, train_labels, labels_name)

    
    
    

# def color(batch):
#     print("Computing descriptor color...")
#     Color = ColorDescriptorExtractor()
#     print("Fitting...")
#     Color.fit(batch)
#     print("Computing...")
#     Color.compute_descriptor(batch)
#     print("Descriptor computed")

#     rfc = RandomForestClassifier(n_estimators=500, max_depth=50, random_state=42)
#     rfc.fit(train_batch, train_labels)

#     plot_results("Color", rfc, train_batch, train_labels, test_batch, labels_test, labels_name)
#     return Color


if __name__ == "__main__":
    __main__()
    