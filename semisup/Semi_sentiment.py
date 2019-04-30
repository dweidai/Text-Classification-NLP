from nltk.tokenize import RegexpTokenizer
#copied from jupyternotebook
tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
middle_words = ['and','a','the','am','it','me','with','in','on','by','near','this','that','an','there','here','those']
middle_words = set(dict.fromkeys([stemmer.stem(word) for word in middle_words]))
def train_classifier(X, y):
    """Train a classifier using the given training data.
        
        Trains logistic regression on the input data with default parameters.
        """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    param_grid = {'C': [0.01, 0.05, 0.1, 0.15, 0.5, 1, 5, 10, 100]}
    grid = GridSearchCV(LogisticRegression(random_state=0, solver='lbfgs',class_weight = 'balanced', max_iter=10000), param_grid, cv=5)
    grid.fit(X, y)
    print("Best cross-validation score: {:.2f}".format(grid.best_score_))
    print("Best parameters: ", grid.best_params_)
    print("Best estimator: ", grid.best_estimator_)
    cls = grid.best_estimator_
    #cls = LogisticRegression(C=0.15, class_weight='balanced', dual=False,
    #      fit_intercept=True, intercept_scaling=1, max_iter=10000,
    #      multi_class='warn', n_jobs=None, penalty='l2', random_state=0,
    #      solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)
    #cls = LogisticRegression(random_state=0, solver='lbfgs', max_iter=10000)
    #cls = LogisticRegression(random_state=0, solver='lbfgs', max_iter=10000)
    cls.fit(X, y)
    return cls

def evaluate(X, yt, cls, name='data'):
    """Evaluated a classifier on the given labeled data using accuracy."""
    from sklearn import metrics
    yp = cls.predict(X)
    acc = metrics.accuracy_score(yt, yp)
    print("  Accuracy on %s  is: %s" % (name, acc))
def read_files(tarfname):
    """Read the training and development data from the sentiment tar file.
        The returned object contains various fields that store sentiment data, such as:
        
        train_data,dev_data: array of documents (array of words)
        train_fnames,dev_fnames: list of filenames of the doccuments (same length as data)
        train_labels,dev_labels: the true string label for each document (same length as data)
        
        The data is also preprocessed for use with scikit-learn, as:
        
        count_vec: CountVectorizer used to process the data (for reapplication on new data)
        trainX,devX: array of vectors representing Bags of Words, i.e. documents processed through the vectorizer
        le: LabelEncoder, i.e. a mapper from string labels to ints (stored for reapplication)
        target_labels: List of labels (same order as used in le)
        trainy,devy: array of int labels, one for each document
        """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    trainname = "train.tsv"
    devname = "dev.tsv"
    for member in tar.getmembers():
        if 'train.tsv' in member.name:
            trainname = member.name
        elif 'dev.tsv' in member.name:
            devname = member.name


class Data: pass
    sentiment = Data()
    print("-- train data")
    sentiment.train_data, sentiment.train_labels = read_tsv(tar,trainname)
    print(len(sentiment.train_data))
    
    print("-- dev data")
    sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)
    print(len(sentiment.dev_data))
    print("-- transforming data and labels")
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    sentiment.count_vect = TfidfVectorizer(ngram_range=(1,3), tokenizer=tokenize) #CountVectorizer()
    sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
    sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)
    from sklearn import preprocessing
    sentiment.le = preprocessing.LabelEncoder()
    sentiment.le.fit(sentiment.train_labels)
    sentiment.target_labels = sentiment.le.classes_
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
    sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
    tar.close()
    return sentiment
def read_unlabeled(tarfname, sentiment):
    """Reads the unlabeled data.
        
        The returned object contains three fields that represent the unlabeled data.
        
        data: documents, represented as sequence of words
        fnames: list of filenames, one for each document
        X: bag of word vector for each document, using the sentiment.vectorizer
        """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    class Data: pass
    unlabeled = Data()
    unlabeled.data = []
    
    unlabeledname = "unlabeled.tsv"
    for member in tar.getmembers():
        if 'unlabeled.tsv' in member.name:
            unlabeledname = member.name

print(unlabeledname)
tf = tar.extractfile(unlabeledname)
for line in tf:
    line = line.decode("utf-8")
    text = line.strip()
    unlabeled.data.append(text)
    
    
    unlabeled.X = sentiment.count_vect.transform(unlabeled.data)
    print(unlabeled.X.shape)
    tar.close()
    return unlabeled
def read_tsv(tar, fname):
    member = tar.getmember(fname)
    print(member.name)
    tf = tar.extractfile(member)
    data = []
    labels = []
    for line in tf:
        line = line.decode("utf-8")
        (label,text) = line.strip().split("\t")
        labels.append(label)
        data.append(text)
    return data, labels
def write_pred_kaggle_file(unlabeled, cls, outfname, sentiment):
    """Writes the predictions in Kaggle format.
        
        Given the unlabeled object, classifier, outputfilename, and the sentiment object,
        this function write sthe predictions of the classifier on the unlabeled data and
        writes it to the outputfilename. The sentiment object is required to ensure
        consistent label names.
        """
    yp = cls.predict(unlabeled.X)
    labels = sentiment.le.inverse_transform(yp)
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    for i in range(len(unlabeled.data)):
        f.write(str(i+1))
        f.write(",")
        f.write(labels[i])
        f.write("\n")
    f.close()
def write_gold_kaggle_file(tsvfile, outfname):
    """Writes the output Kaggle file of the truth.
        
        You will not be able to run this code, since the tsvfile is not
        accessible to you (it is the test labels).
        """
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    i = 0
    with open(tsvfile, 'r') as tf:
        for line in tf:
            (label,review) = line.strip().split("\t")
            i += 1
            f.write(str(i))
            f.write(",")
            f.write(label)
            f.write("\n")
    f.close()
def write_basic_kaggle_file(tsvfile, outfname):
    """Writes the output Kaggle file of the naive baseline.
        
        This baseline predicts POSITIVE for all the instances.
        """
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    i = 0
    with open(tsvfile, 'r') as tf:
        for line in tf:
            (label,review) = line.strip().split("\t")
            i += 1
            f.write(str(i))
            f.write(",")
            f.write("POSITIVE")
            f.write("\n")
    f.close()

#semi supervised code here
import numpy as np
from scipy.sparse import vstack
def semi_supervised(cls):
    print("\nReading unlabeled data")
    unlabeled = read_unlabeled(tarfname, sentiment)
    print("\nTraining semi-supervised classifier")
    print()
    confident = 0.4 # will be used to set threshold on the values to extend or not
    percent = 1 #how much of the unlabeled data do we want
    length = len(unlabeled.data)* percent
    checked = dict() #keep track of the data that we added, so no redundancy
    restriction = 500 #extended number of words lower limit
    previous = 0
    init = 0 #training
    while (len(checked)-previous <= restriction):
        #predict is the number of data we want from unlabeled
        #we picked all of them
        predict = cls.predict_proba(unlabeled.X[:int(length),:])
        print(confident_labels.shape)
        labels = np.zeros((len(predict), 3))
        print(labels.shape)
        #creat a seperatearray to track the numbers
        for index, p in enumerate(predict):
            labels[index][0] = index
            labels[index][1] = p[0]
            labels[index][2] = p[1]
        print("labels created")
        print("semi supervised training starts")
        for i in range(labels.shape[0]):
            predict = -1
            if labels[i][0] in checked:
                print(labels[i][0])
                continue
            # if they are bigger than confident, we use them
            elif labels[i][1] > confident:
                predict = cls.predict(unlabeled.X[labels[i][0]])
                checked[labels[i][0]] = True
            elif labels[index][2] > confident:
                predict = cls.predict(unlabeled.X[labels[i][0]])
                checked[labels[i][0]] = True
            if predict >= 0: #expand the labels
                #here we are going to implement them to the extension of the data
                sentiment.trainX = vstack([sentiment.trainX, unlabeled.X[labels[i][0]]])
                sentiment.trainy = np.concatenate((sentiment.trainy, predict))
                print(sentiment.trainy.shape)
        print(len(checked), previous)
        #update the previous value to keep while loop
        previous = len(checked)
        #train int the end with newer extended data set
        cls = train_classifier(sentiment.trainX, sentiment.trainy)
    print("evaluating")
    evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')
    acc = evaluate(sentiment.devX, sentiment.devy, cls, 'dev')
    print("Writing predictions to a file")
    unlabeled = read_unlabeled(tarfname, sentiment)
    write_pred_kaggle_file(unlabeled, cls, "data/sentiment-pred.csv", sentiment)
