import glob
import gensim
from gensim.utils import simple_preprocess
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
import nltk
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import wordnet as wn
import pyLDAvis.gensim

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

directory_path = "./Chapters/"

text_files = glob.glob(f"{directory_path}/*.txt")
text_files = [file for file in text_files if "CHAPTER" in file or "Epilogue" in file]
chapters = []


nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()


for file in text_files:
    text = open(file, encoding="utf-8").read()
    text = simple_preprocess(text)
    #chapters.append(text)
    
    #print(text)
    nltk_tagged = nltk.pos_tag(text)
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        #print(word, tag)
        if tag is not None:
            # If there is no tag, ignore the word
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    #print(lemmatized_sentence)
    chapters.append(lemmatized_sentence)
    

common_dictionary = Dictionary(chapters)
common_corpus = [common_dictionary.doc2bow(text) for text in chapters]
lda = LdaModel(common_corpus, id2word=common_dictionary)

#print(lda.print_topics())
#pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda, common_corpus, dictionary=lda.id2word)
pyLDAvis.save_html(vis, 'LDA_Visualization.html') 