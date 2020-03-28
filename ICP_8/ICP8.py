import json
from stanfordcorenlp import StanfordCoreNLP

# Preset
host = 'http://localhost'
port = 9000
nlp = StanfordCoreNLP(host, port=port,timeout=15000)

#nlp=StanfordCoreNLP("http://localhost:9000/")
s='Linda is PhD student . she is studying at the University of Missouri Kansas City . she is doing her PhD in the area of natural language processing . she has a friend called Brenda who is an MSc student in the same College. Linda also works a grader in the University. There are a few other academic staff who work along with Linda.'
s1='A Gene, which is a short form of genetic material acts as a gourd creating proteins human body. a mutation in gene is the real cause of a disease. a mutation in the gene affect proteins in the body which cause disease. there is a lot of research going on about identifying diseases and the corresponding proteins affected. the diseases and their corresponding proteins are cited in a lot of papers and these are called citations. a great deal of details about genes proteins and diseases are discussed in the citations. Citations are an essential part of the academic research.'
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

# importing the library
from nltk.corpus import wordnet

# lets use word paint as an exqmple
w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('boat.n.01')

print(sorted(w1.common_hypernyms(w2)))