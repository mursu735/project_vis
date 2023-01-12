import helpers
import gensim.models
import opennre
'''
model_name = helpers.fetch_doc2vec_model_name()

new_model = gensim.models.Doc2Vec.load(f"{model_name}")

#print(new_model.dv['Chapter 1/paragraph 1'])

print(new_model.dv.get_vector('Chapter 1/paragraph 1'))

tags = []

with open("doc2vec_tags.txt") as file:
    tags = [line.replace('\n','') for line in file.readlines()]

print(tags)'''

model = opennre.get_model('wiki80_cnn_softmax')
asd = model.infer({'text': 'He was the son of Máel Dúin mac Máele Fithrich, and grandson of the high king Áed Uaridnach (died 612).', 'h': {'pos': (18, 46)}, 't': {'pos': (78, 91)}})
print(asd)