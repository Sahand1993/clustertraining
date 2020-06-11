import json
import pickle

failed_by_dssm_only = pickle.load(open("failed_by_dssm_only", "rb"))

mid = open("datasets_titlenq/nq/mid.jsonl")

idswherebm25wins = set()
for jsonLine in mid:
    jsonobj = json.loads(jsonLine)

    if int(jsonobj["exampleId"]) in failed_by_dssm_only:
        print(jsonobj["questionTokens"], jsonobj["documentTitleTokens"])
        idswherebm25wins.add(jsonobj["exampleId"])

winbm25json = open("datasets_titlenq/nq/test_bm25_wins.jsonl", "w")
winbm25csv = open("datasets_titlenq/nq/test_bm25_wins.csv", "w")

testjson = open("datasets_titlenq/nq/test.jsonl")
testcsv = open("datasets_titlenq/nq/test.csv")
testcsv.readline()
for jsonline, csvline in zip(testjson, testcsv):
    idcsv = int(csvline.split(";")[1])
    idjson = json.loads(jsonline)["exampleId"]
    if idcsv != idjson:
        raise ValueError("ids not the same")
    if idcsv in idswherebm25wins:
        winbm25json.write(jsonline)
        winbm25csv.write(csvline)
