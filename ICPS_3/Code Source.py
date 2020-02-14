import json
from stanfordcorenlp import StanfordCoreNLP

# Preset
host = 'http://localhost'
port = 9000
nlp = StanfordCoreNLP(host, port=port,timeout=15000)

#nlp=StanfordCoreNLP("http://localhost:9000/")
s='The biggest natural disaster in 2020 was the burning of the Amazon rainforest fire.'
s1='The biggest natural disaster in 2020 was Coronavirus spread in China.'
s = input('Sentence: ')
output = nlp.annotate(s1, properties={"annotators":"tokenize,ssplit,pos,depparse,natlog,openie",
                                "outputFormat": "json",
                                 "openie.triple.strict":"true",
                                 "openie.max_entailments_per_clause":"1"})
a = json.loads(output)
print("The subject, object and verb/relation of the given sentence are")
print(a["sentences"][0]["openie"],'\n')
result = [a["sentences"][0]["openie"] for item in a]
for i in result:
    for rel in i:
        relationSent=rel['relation'],rel['subject'],rel['object']
        print('The triplet of the given sentence is')
        print(relationSent)

# command to import wordnet library from nltk
from nltk.corpus import wordnet

# using some exqmples as an w1 & w2
w1 = wordnet.synset('man.n.01')
w2 = wordnet.synset('boy.n.01')

print(sorted(w1.common_hypernyms(w2)))


