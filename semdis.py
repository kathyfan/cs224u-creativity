import scipy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk


# takes two words and return their semantic distance in a VSM
def get_word_cosine(word1, word2, vsm):
  v1 = vsm[word1],
  v2 = vsm[word2]

  return(scipy.spatial.distance.cosine(v1, v2))



def normalized_tfidf(df):
    vec = CountVectorizer(tokenizer= nltk.word_tokenize,
                          stop_words = {'english'})
    dtf = vec.fit_transform(df['text']).toarray()

    tfidf_vec = TfidfVectorizer(tokenizer= nltk.word_tokenize,
                          stop_words = {'english'},
                               use_idf=False)
    tfidf = tfidf_vec.fit_transform(df['text']).toarray()
    
    normed_tfidf= tfidf.sum(axis = 1)/dtf.sum(axis = 1)
    return normed_tfidf