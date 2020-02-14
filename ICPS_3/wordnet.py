# importing the library
from nltk.corpus import wordnet

# using example word
syns = wordnet.synsets("bird")

# synset example 
print(syns[0].name())
print('\n')

# only word:
print(syns[0].lemmas()[0].name())
print('\n')

# 1st synset definition
print(syns[0].definition())
print('\n')

# using example word in sentences:
print(syns[0].examples())
print('\n')

# synonyms with antonyms in wordnet use the word
antonyms = []
synonyms = []


for syn in wordnet.synsets("girl"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())
print('The synonyms of girl is: ')
print(set(synonyms))
print('\n')
print('The antonyms of girl is: ')
print(set(antonyms))
print('\n')


# the two words similarity are:
w1 = wordnet.synset('man.n.01')
w2 = wordnet.synset('boy.n.01') # n denotes noun
print("Similarity betwee man & boy is =",w1.wup_similarity(w2))