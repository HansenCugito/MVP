#!/usr/bin/python

import random
import collections
import copy
import math
import sys
import os, random, operator, sys
from collections import Counter

def extractWordFeatures(x):
    #Feature Extractor No.1
    #Train Error: 2.2% Test Error: 26.73%
    #return word features of a string delimited by whitespace characters only.
    #Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    featurevector=collections.defaultdict(int)
    for features in x.split(): featurevector[features]+=1
    return featurevector

def extractCharacterFeatures(n):
    #Feature Extractor No.2
    #Train Error: 0.0% Test Error:27.07%
    #return a feature extractor consisting of all n-grams of a string without spaces
    #EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    #in this MRP project, n=5 produces the smallest error
    def extract(x):
        text=x.replace(" ","")
        textlist=[text[i:i+n] for i in range(len(text)-n+1)]
        return collections.Counter(textlist)
    return extract

def getMargin(weights, feature, y):
    # y is the label, the value is {-1, 1}.
    return dotProduct(weights, feature) * y

def sparseVectorMultiplication(v, scale) :
    for key in v:
        v[key] = v[key] * scale

def sgd(weights, feature, label, eta):
    # Updates weight.
    # label has value {-1, 1}.
    gradient = collections.defaultdict(float)
    if (1 - getMargin(weights, feature, label)) > 0 :
        gradient = feature
        sparseVectorMultiplication(gradient, -label)
    else:
        # gradient is all 0 in this case.
        pass

    increment(weights, (-1) * eta, gradient)

def getPredictor(weights, featureExtractor):
    # A linear -1/1 predictor, the decision boundary is 0.
    return lambda x : 1 if dotProduct(weights, featureExtractor(x)) >= 0 else -1

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned by implementing stochastic gradient descent.
    EvaluatePredictor() is called on both trainExamples and testExamples
    to see how we learn after each iteration.
    '''
    weights = {}  
    for i in range(numIters):
        trainSamples=random.sample(trainExamples,len(trainExamples))
        for (x, y) in trainSamples:
            sgd(weights, featureExtractor(x), y, eta)
        for (k, v) in weights.items(): weights[k]=round(weights[k],2)
        evaluatetrain=evaluatePredictor(trainExamples,getPredictor(weights, featureExtractor))
        evaluatetest=evaluatePredictor(testExamples,getPredictor(weights, featureExtractor))
    return weights

def generateDataset(numExamples, weights):
    #Return a set of examples (phi(x), y) randomly which are classified correctly by |weights|.
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) is a dict whose keys are a subset of the keys in weights
    # and values can be anything with a nonzero score under the given weight vector.
    # y is 1 or -1 as classified by the weight vector.
    def generateExample():
        weights_keys=[k for (k,v) in weights.items()]
        phi={random.choice(weights_keys) : random.randrange(1,5,1) for ele in range(len(weights_keys))}
        y=1 if dotProduct(weights, phi) >= 0 else -1
        return (phi, y)

    return [generateExample() for _ in range(numExamples)]


### utility functions ###

def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())

def increment(d1, scale, d2):
    #Implements d1 += scale * d2 for sparse vectors.
    for f, v in d2.items():
        d1[f] = d1.get(f, 0) + v * scale

def readExamples(path):
    #Reads a set of training examples. 
    examples = []
    for line in open(path):
        # Format of each line: <output label (+1 or -1)> <input sentence>
        y, x = line.split(' ', 1)
        examples.append((x.strip(), int(y)))
    print( 'Read %d examples from %s' % (len(examples), path))
    return examples

def evaluatePredictor(examples, predictor):
    '''
    predictor: a function that takes an x and returns a predicted y.
    Given a list of examples (x, y), makes predictions based on |predict| and returns the fraction
    of misclassiied examples.
    '''
    error = 0
    for x, y in examples:
        if predictor(x) != y:
            error += 1
    return 1.0 * error / len(examples)

def outputErrorAnalysis(examples, featureExtractor, weights, path):
    out = open('error-analysis', 'w')
    for x, y in examples:
        out.write('===', x)
        verbosePredict(featureExtractor(x), y, weights, out)
    out.close()

def interactivePrompt(featureExtractor, weights):
    while True:
        print ('> ',)
        x = sys.stdin.readline()
        if not x: break
        phi = featureExtractor(x) 
        verbosePredict(phi, None, weights, sys.stdout)

############################################

#Algorithm to train Weights given small dataset, obtain trained weights,
#generate large datasets with trained weights, train weights with large datasets#
        
#start training and obtaining weights from given trainExamples and testExamples
weights=learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta)
#generate dataset phi(x),y by using the weights learned
generatedDataset=generateDataset(numExamples, weights) 
#now merge the datasets to obtain Examples
merge=[tuple() for i in range(numExamples)]
i=0
for (phi,y) in generatedDataset:
    example=[(k+' ')*v for(k,v) in phi.items()]
    merge[i]=(''.join(example),y)
    i+=1
#after n examples are generated, allocate 10% to testExamples and remaining 90% to trainExamples
generated_trainExamples=[tuple() for i in range(int(numExamples*9/10))]
generated_testExamples=[tuple() for i in range(int(numExamples*1/10))]
n=0
for(x, y) in generatedDataset:
    if n<int(numExamples*9/10): 
        generated_trainExamples[n]=merge[n]
    else: 
        generated_testExamples[n-int(numExamples*9/10)]=merge[n]
    n+=1
#Finally train the weights with the generated examples
weights=learnPredictor(generated_trainExamples, generated_testExamples, featureExtractor, numIters, eta)
