import json
from nltk import ngrams
from nltk import bigrams
import nltk
import pickle
from scipy.spatial import distance
import os

category = ["wannacry", "cerber", "ctb", "gandcrab", "locky"]
binaryApiText = ""

validate = False
reason = ""

checkFileList = ["hello.docx","hello.hwp","hello.png","hello.pptx","hello.txt","hello.xlsx"]    # 확장자 변경 탐지

with open("./wannacry/reports2/report.json") as report: # 분석할 바이너리 report.json 경로
    data = json.load(report)
    
    if(data["behavior"]["summary"].get("file_moved")):
                for k in data["behavior"]["summary"]["file_moved"]:  
                    beforePath, beforeExtension = os.path.splitext(k[0])
                    if(os.path.basename(beforePath) in checkFileList):
                        validate = True
                        reason = "파일의 확장자를 변경하는 행위가 발견"

    for i in data["behavior"]["processes"]:
        if(data["target"]["file"]["name"] in i["process_name"] ):
            for j in i["calls"]:
                binaryApiText = binaryApiText + j["api"] + " "
            break

binanryApiList = nltk.word_tokenize(binaryApiText) # 공백으로 분리되어 있는 String들을 한 단어 씩 묶어서 리스트로 
twoGramList = list(ngrams(binanryApiList,2)) # 2 gram
apiCountData = [(item, twoGramList.count(item)) for item in set(twoGramList)]

similarList = []

for category in category:

    common_gram = []
    binaryDataSet = []
    sampleDataSet = []
    binaryCheck = False
    sampleCheck = False
    

    with open(f"./sample/2-gram/{category}.pkl","rb") as f:
        sampleCountData = pickle.load(f)
        common_gram = set([(item) for item in set(tuple(sampleCountData))]) | set([item for item in set(tuple(apiCountData))])
        common_gram = list(common_gram)

    for i in range(0,len(common_gram)):
        for j in range(0,len(apiCountData)):
            if(apiCountData[j][0] == common_gram[i][0]):
                
                binaryDataSet.append(apiCountData[j][1])
                binaryCheck = True
                break

        if(binaryCheck != True):
            binaryDataSet.append(0)

        for k in range(0,len(sampleCountData)):
            if(sampleCountData[k][0] == common_gram[i][0]):
                
                sampleDataSet.append(sampleCountData[k][1])
                sampleCheck = True
                break

        if (sampleCheck != True):
            sampleDataSet.append(0)

        binaryCheck = False
        sampleCheck = False
    print(category, distance.braycurtis(sampleDataSet,binaryDataSet))   # scipy distance
    similarList.append(distance.braycurtis(sampleDataSet,binaryDataSet))    

for i in similarList:
    if(i < 0.4):
        validate = True
        reason = reason + ". 랜섬웨어 샘플과의 유사도 비교에서 높은 유사도를 보임"

if(validate):
    print(reason)   
else:
    print("탐지 실패 or 정상파일")




