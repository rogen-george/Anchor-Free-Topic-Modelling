
import nltk
from nltk.corpus import reuters
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import cvxopt
import numpy
import string
from cvxpy import *
from scipy.sparse.linalg import eigs
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np

import os

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems

# Number of training documents
size = 1440
vocab = 19316

train_documents = []
topics = ["business", "entertainment", "politics", "tech"]
k = len(topics);

import time
start = time.time()
#your code here

for topic in topics:
    for filename in os.listdir("data4/" + topic):

        if filename.endswith(".txt"):
            f = open("data4/" + topic + "/" + filename, "r")
            document = f.read().decode('utf-8', errors='ignore')
            exclude = set(string.punctuation)
            document = ''.join(ch for ch in document if ch not in exclude)
            train_documents.append(document)

vectorizer = TfidfVectorizer(tokenizer = tokenize, stop_words = 'english')

newArray = [train_documents[i] for i in range(size)]
vectorised_train_documents = vectorizer.fit_transform(newArray)

# Prints the vector to vocabulary mapping
values = vectorizer.get_feature_names()

print("Completed computation of Co-occurrence matrix.")

co_matrix = vectorised_train_documents.T * vectorised_train_documents
#co_matrix.setdiag(0)
co_matrix = co_matrix / size

# Applying Eigen decomposition to D matrix
print("Applying square root decomposition to Co-occurrence matrix... ")

vals, vecs = eigs(co_matrix, k=4)

vals_real = numpy.real(vals)
vals_d = numpy.sqrt(vals_real)
d_root = numpy.diag(vals_d)

Bo = numpy.real(numpy.matmul(vecs, d_root))
ones = numpy.ones(vocab)

# Initialise M here, definitely a different method to compute M here
M = numpy.identity(k)

# Printing the top 20 words in each category


for _ in range(10):


    for i in range(0, k):

        a = numpy.zeros(k)

        # Computing a here
        for j in range(0, k):
            step = (-1.)**(i+j)
            N = numpy.delete(M, j, axis=0)
            N = numpy.delete(N, i, axis=1)
            Nt = numpy.linalg.det(N)
            a[j] = (step*Nt)

        # Solving the optimization problem
        x = Variable(k)

        Bo_sum = numpy.sum( Bo, axis = 0)
        ineq = ( Bo * x >= 0 )
        ineq2 = ( Bo_sum * x == 1)

        # build optimization problem
        prob_m1 = Problem( Maximize(a * x), [ ineq, ineq2 ])
        argmax = prob_m1.solve()

        y = Variable(k)
        ineq3 = ( Bo * y >= 0 )
        ineq4 = ( Bo_sum * y == 1)

        prob_m2 = Problem( Minimize(a * y), [ ineq3, ineq4 ])

        # solve optimization problem and prints results
        argmin = prob_m2.solve()

        if ( numpy.linalg.norm(a * x.value) ) >= ( numpy.linalg.norm(a * y.value) ):
            final = x.value
        else:
            final = y.value

        M[:,i] = final   # check arguments here

C = numpy.matmul(Bo, M)  #matrix multiplication logic here, final C
Ct = C.transpose()

C_new = C.T

# Printing the top 30 words in each category
for k in range(4):
    print (k)
    words = C_new[k].argsort()[-30:][::-1]
    print words
    for word in words:
        print values[word]

print time.time() - start
