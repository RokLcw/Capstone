import json
from nltk import ngrams
from nltk import bigrams
import nltk
import pickle
from scipy.spatial import distance
import os

category = ["cerber","ctb","locky","wannacry"]
binaryApiText = ""

validate = False
reason = ""

with open("./wannacry/reports1/report.json") as report: # 분석할 바이너리 report.json 경로
    data = json.load(report)

    for i in data["behavior"]["processes"]:
        if(data["target"]["file"]["name"] in i["process_name"] ):
            for j in i["calls"]:
                binaryApiText = binaryApiText + j["api"] + " "
            break

binanryApiList = nltk.word_tokenize(binaryApiText) # 공백으로 분리되어 있는 String들을 한 단어 씩 묶어서 리스트로 
twoGramList = list(ngrams(binanryApiList,4)) # 2 gram
apiCountData = [(item, twoGramList.count(item)) for item in set(twoGramList)]

with open('wannacry_n4_test.pkl', 'wb') as f:
    pickle.dump(apiCountData, f, pickle.HIGHEST_PROTOCOL)