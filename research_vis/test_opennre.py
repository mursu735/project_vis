import helpers
#import gensim.models
#import opennre
import pandas as pd
'''
model_name = helpers.fetch_doc2vec_model_name()

new_model = gensim.models.Doc2Vec.load(f"{model_name}")

#print(new_model.dv['Chapter 1/paragraph 1'])

print(new_model.dv.get_vector('Chapter 1/paragraph 1'))

tags = []

with open("doc2vec_tags.txt") as file:
    tags = [line.replace('\n','') for line in file.readlines()]

print(tags)'''
'''
model = opennre.get_model('wiki80_bert_softmax')
text = "But I had not proceeded far, when I began to bethink me that the Captain with whom I was to sail yet remained unseen by me; though, indeed, in many cases, a whale-ship will be completely fitted out, and receive all her crew on board, ere the captain makes himself visible by arriving to take command; for sometimes these voyages are so prolonged, and the shore intervals at home so exceedingly brief, that if the captain have a family, or any absorbing concernment of that sort, he does not trouble himself much about his ship in port, but leaves her to the owners till all is ready for sea. However, it is always as well to have a look at him before irrevocably committing yourself into his hands. Turning back I accosted Captain Peleg, inquiring where Captain Ahab was to be found. "
target2 = "Ahab"
target1 = "Peleg"


start_h = text.index(target1)
end_h = start_h + len(target1)

start_t = text.index(target2)
end_t = start_t + len(target2)

print(start_h, end_h)

print(start_t, end_t)

asd = model.infer({'text': text, 
    'h': {'pos': (start_h, end_h)}, 't': {'pos': (start_t, end_t)}})
print(asd)
'''
asd = pd.read_json("test.json", convert_dates=False)
print(asd.keys())