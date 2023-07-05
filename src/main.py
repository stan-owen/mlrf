from color import *
from bovw import *
from flat import *
from data import *
from result import *
from classifieur import *

def __main__():
    bovw()
    return True
    
def bovw():
    train_batch, train_labels, test_batch, test_labels, labels_name = load_data()

    Bovw_train = BOVWDescriptorExtractor("bovw_train")
    Bovw_train.fit(train_batch)
    Bovw_train.compute_descriptor(train_batch)

    Bovw_test = BOVWDescriptorExtractor("bovw_test")
    Bovw_test.fit(test_batch)
    Bovw_test.compute_descriptor(test_batch)

    Classifieur = ClassifieurDescriptor("BOVW", Bovw_train.image_descriptor, train_labels, None, None, None)

    Classifieur.fit_rfc()
    plot_results(Classifieur.name, Classifieur.rfc, Bovw_test.image_descriptor, test_labels, labels_name)

    # Classifieur.fit_logReg()
    # plot_results(Classifieur.name, Classifieur.logReg, Bovw_test.image_descriptor, test_labels, labels_name)

    # Classifieur.fit_svc()
    # plot_results(Classifieur.name, Classifieur.svc, Bovw_test.image_descriptor, test_labels, labels_name)

    

def color():

    train_batch, train_labels, test_batch, test_labels, labels_name = load_data()

    Color_train = ColorDescriptorExtractor("color_train")
    Color_train.fit(train_batch)
    Color_train.compute_descriptor(train_batch)

    Color_test = ColorDescriptorExtractor("color_test")
    Color_test.fit(test_batch)
    Color_test.compute_descriptor(test_batch)

    Classifieur = ClassifieurDescriptor("Color", Color_train.histogram, train_labels, None, None, None)

    Classifieur.fit_rfc()
    plot_results(Classifieur.name, Classifieur.rfc, Color_test.image_descriptor, test_labels, labels_name)

    # Classifieur.fit_logReg()
    # plot_results(Classifieur.name, Classifieur.logReg, Color_test.image_descriptor, test_labels, labels_name)

    # Classifieur.fit_svc()
    # plot_results(Classifieur.name, Classifieur.svc, Color_test.image_descriptor, test_labels, labels_name)

def flat():
    train_batch, train_labels, test_batch, test_labels, labels_name = load_data()

    Flat_train = FlatDescriptorExtractor("flat_train")
    Flat_train.compute_descriptor_flat(train_batch)

    Flat_test = FlatDescriptorExtractor("flat_test")
    Flat_test.compute_descriptor_flat(test_batch)

    Classifieur = ClassifieurDescriptor("Flat", Flat_train.train_data, train_labels, None, None, None)

    Classifieur.fit_rfc()
    plot_results(Classifieur.name, Classifieur.rfc, Flat_test.train_data, test_labels, labels_name)

    # Classifieur.fit_logReg()
    # plot_results(Classifieur.name, Classifieur.logReg, Flat_test.train_data, test_labels, labels_name)

    # Classifieur.fit_svc()
    # plot_results(Classifieur.name, Classifieur.svc, Flat_test.train_data, test_labels, labels_name)


if __name__ == "__main__":
    __main__()
    