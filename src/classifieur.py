import joblib

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier



class ClassifieurDescriptor:
    def __init__(self, name, batch, labels, rfc=None, logReg=None, svc=None):
        self.name = name
        self.batch = batch
        self.labels = labels
        self.rfc = None if rfc is None else joblib.load(rfc)
        self.logReg = None if logReg is None else joblib.load(logReg)
        self.svc = None if svc is None else joblib.load(svc)

    def fit_rfc(self):
        print(self.name + " : Fitting... Random Forest Classifier")
        rfc = RandomForestClassifier(n_estimators=500, max_depth=50, random_state=42)
        rfc.fit(self.batch, self.labels)
        self.rfc = rfc
        joblib.dump(rfc, self.name + ' _rfc.joblib')
        print(self.name + " : Random Forest Classifier done")

    def fit_logReg(self):
        print(self.name + " : Fitting... Logistic Regression")
        logReg = LogisticRegression(c=10, penalty='l2', random_state=42)
        logReg.fit(self.batch, self.labels)
        self.logReg = logReg
        joblib.dump(logReg, self.name + ' _logReg.joblib')
        print(self.name + " : Logistic Regression done")

    def fit_svc(self):
        print(self.name + " : Fitting... SVC")
        svc = SVC(random_state=42, probability=True)
        svc.fit(self.batch, self.labels)
        self.svc = svc
        joblib.dump(svc, self.name + ' _svc.joblib')
        print(self.name + " : SVC done")
