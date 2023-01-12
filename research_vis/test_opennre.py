import helpers
#import gensim.models
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

model = opennre.get_model('wiki80_bert_softmax')
text = "Next was Tashtego, an unmixed Indian from Gay Head, the most westerly promontory of Martha’s Vineyard, where there still exists the last remnant of a village of red men, which has long supplied the neighboring island of Nantucket with many of her most daring harpooneers. In the fishery, they usually go by the generic name of Gay-Headers. Tashtego’s long, lean, sable hair, his high cheek bones, and black rounding eyes—for an Indian, Oriental in their largeness, but Antarctic in their glittering expression—all this sufficiently proclaimed him an inheritor of the unvitiated blood of those proud warrior hunters, who, in quest of the great New England moose, had scoured, bow in hand, the aboriginal forests of the main. But no longer snuffing in the trail of the wild beasts of the woodland, Tashtego now hunted in the wake of the great whales of the sea; the unerring harpoon of the son fitly replacing the infallible arrow of the sires. To look at the tawny brawn of his lithe snaky limbs, you would almost have credited the superstitions of some of the earlier Puritans, and half-believed this wild Indian to be a son of the Prince of the Powers of the Air. Tashtego was Stubb the second mate’s squire."
target1 = "Tashtego"
target2 = "Stubb"


start_h = text.index(target1)
end_h = start_h + len(target1)

start_t = text.index(target2)
end_t = start_t + len(target2)

print(start_h, end_h)

print(start_t, end_t)

asd = model.infer({'text': text, 
    'h': {'pos': (start_h, end_h)}, 't': {'pos': (start_t, end_t)}})
print(asd)