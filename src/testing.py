from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import pandas as pd
import gensim
from gensim import corpora
from nltk.corpus import stopwords
import spacy


data = pd.read_csv("src/data.csv")
dataFiltered = data["content"][:100]

stop_words = stopwords.words('english')

# function to remove stopwords
def remove_stopwords(text):
    textArr = text.split(' ')
    rem_text = " ".join([i for i in textArr if i not in stop_words])
    return rem_text

# remove stopwords from the text
dataFiltered=dataFiltered.apply(remove_stopwords)

# nlp = en_core_web_sm.load()
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def lemmatization(texts,allowed_postags=['NOUN', 'ADJ']): 
       output = []
       for sent in texts:
             doc = nlp(sent) 
             output.append([token.lemma_ for token in doc if token.pos_ in allowed_postags ])
       return output

text_list=dataFiltered.tolist()
print(text_list[5])
tokenized_reviews = lemmatization(text_list)
print(tokenized_reviews[5])

# -- Omar was here --------------------------------------------------------------------------------

dictionary = corpora.Dictionary(tokenized_reviews)

doc_term_matrix = [dictionary.doc2bow(doc) for doc in tokenized_reviews]

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(corpus=doc_term_matrix, id2word=dictionary, num_topics=20, random_state=100)
print("-"*10, "\n", "Topics\n", ldamodel.print_topics())

#visualize the topics
#pyLDAvis.enable_notebook()

# Visualize the topics
vis = pyLDAvis.gensim.prepare(ldamodel, doc_term_matrix, dictionary)
pyLDAvis.save_html(vis, 'src/test.html')
vis

# # Tokenize the data
# tokenized_data = [gensim.utils.simple_preprocess(str(i)) for i in dataFiltered]

# # Create a dictionary representation of the documents
# dictionary = Dictionary(tokenized_data)

# # Create a bag-of-words representation of the documents
# corpus = [dictionary.doc2bow(i) for i in tokenized_data]

# # Create a LDA model
# lda_model = LdaModel(corpus=corpus,
#                      id2word=dictionary,
#                      num_topics=20,
#                      random_state=100)

# # Print the coherence score
# if __name__ == '__main__': 
#     # Print the topics
#    # print("-"*10, "\n", "Topics\n", lda_model.print_topics())

#     coherence_model_lda = CoherenceModel(
#         model=lda_model, texts=tokenized_data, dictionary=dictionary, coherence='c_v')
#     print("-"*10, "\n", "Coherence\n", coherence_model_lda.get_coherence())

#     # Print the preplexity
#     print("-"*10, "\n", "Preplexity\n", lda_model.log_perplexity(corpus))
