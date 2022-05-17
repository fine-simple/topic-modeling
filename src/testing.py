from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import pandas as pd
import gensim

data = pd.read_csv("articles1.csv")
dataFiltered = data["content"][:1000]

# Topic Modelling using gensim
# Tokenize the data
tokenized_data = [gensim.utils.simple_preprocess(str(i)) for i in dataFiltered]

# Create a dictionary representation of the documents
dictionary = Dictionary(tokenized_data)

# Create a bag-of-words representation of the documents
corpus = [dictionary.doc2bow(i) for i in tokenized_data]

# Create a LDA model
lda_model = LdaModel(corpus=corpus,
                     id2word=dictionary,
                     num_topics=10,
                     random_state=100)

# Print the coherence score
if __name__ == '__main__':
    # Print the topics
    print("-"*10, "\n", "Topics\n", lda_model.print_topics())

    coherence_model_lda = CoherenceModel(
        model=lda_model, texts=tokenized_data, dictionary=dictionary, coherence='c_v')
    print("-"*10, "\n", "Coherence\n", coherence_model_lda.get_coherence())

    # Print the preplexity
    print("-"*10, "\n", "Preplexity\n", lda_model.get_perplexity(corpus))
