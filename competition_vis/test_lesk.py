import nltk
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from pywsd.lesk import adapted_lesk, simple_lesk

import time

def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

start_time = time.time()

lemmatizer = WordNetLemmatizer()

sent = "Trucks are on fire"
#"such bad traffic... Trucks are on fire."
word = 'fire'

tokenized = nltk.word_tokenize(sent)
#print(tokenized)

for ss in wn.synsets(word):
    print(ss, ss.definition(), "n")
'''
nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sent))
#tuple of (token, wordnet_tag)
wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
lemmatized_sentence = []
for word, tag in wordnet_tagged:
    print(word, tag)
    if tag is None:
        #if there is no available tag, append the token as is
        lemmatized_sentence.append(word)
    else:
        #else use the tag to lemmatize the token
        lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))

print(lemmatized_sentence)
'''
asd = adapted_lesk(sent, word)
print(asd)
print(f"Time elapsed: {time.time() - start_time}")

