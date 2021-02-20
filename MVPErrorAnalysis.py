#!/usr/bin/python

import util
import time
from util import *

############################################################
# Problem 3: sentiment classification

class File:
    def load(self, moduleName):
        try:
            return __import__(moduleName)
        except Exception as e:
            self.fail("Threw exception when importing '%s': %s" % (moduleName, e))
            self.fatalError = True
            return None
        except:
            self.fail("Threw exception when importing '%s'" % moduleName)
            self.fatalError = True
            return None
        
MVP = File.load('MovieRatingsPredictor')

def ErrorAnalysis():
    trainExamples = readExamples('polarity.train')
    testExamples = readExamples('polarity.dev')
    featureExtractor = submission.extractWordFeatures
    weights = submission.learnPredictor(trainExamples, testExamples, featureExtractor, numIters=20, eta=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(testExamples, featureExtractor, weights, 'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(trainExamples, lambda x : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    devError = evaluatePredictor(devExamples, lambda x : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print( "Official: train error = %s, dev error = %s" % (trainError, devError))
