import scipy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
import scipy.spatial.distance


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


# Takes two sequence and create a composite vector for each sequence.
# Return the cosine similarity between the two vectors.
# The multiply argument regulates whether mulplicative or additive composition is used
def get_distance_between_texts(text1, text2, VSM = glove_dict,
                               multiply = True,
                               tokenizer = nltk.word_tokenize,
                               remove_stopwords = True,
                               remove_punct = True):
  
  v1 = get_text_vector(text1, VSM, multiply, tokenizer, remove_stopwords),
  v2 = get_text_vector(text2, VSM, multiply, tokenizer, remove_stopwords)

  return scipy.spatial.distance.cosine(v1, v2)

# Takes a sequence and a VSM. Return a composite vector that represents the sequence
# Extract word vectors from the VSM and combine them with either multiplication or addition (default is multiplication)
# Set multiply = False to use addition
# Default tokenizer is nltk word tokenizer. 
# Remove stopwords and punctuations by default.

## TODO: Trying weighted sum (e.g., IDF weighting)
def get_text_vector(text, 
                    VSM, # the VSM (a dictionary) used to derive word vectors
                    multiply = True,
                    tokenizer = nltk.word_tokenize,
                    remove_stopwords = True,
                    remove_punct = True):
  
  if remove_punct:
    text = text.translate(str.maketrans('','',string.punctuation))
  
  
  words = tokenizer(text)

  if remove_stopwords:
    stop_words = nltk.corpus.stopwords.words('english')
    words = [w for w in words if not w in stop_words] 

    
  
  words = [w for w in words if w in VSM] 

  # Uncomment this for sanity check
  #print(len(words))
  if len(words) > 0:
    v = VSM[words[0]]
    for word in words[1:]:
      if multiply:
        v= np.multiply(v, VSM[word])
      else:
        v = v+VSM[word]
  else:
    # If no word is found in the dictionary, return a random vector
    v = np.random.rand(300)

  return v

# take a sentence and return the semantic distances between 
def distances_within_text(text,
                          tokenizer = nltk.word_tokenize,
                         remove_stopwords =True,
                             remove_punct = True):
    if remove_punct:
        text = text.translate(str.maketrans('','',string.punctuation))
    words = tokenizer(text)
    if remove_stopwords:
        stop_words = nltk.corpus.stopwords.words('english')
        words = [w for w in words if not w in stop_words] 
    n = len(words)
    distances = []
    for i in range(n):
        for j in range(i+1, n):
            if words[i] in glove_dict and words[j] in glove_dict:
                distances.append(semdis.get_word_cosine(words[i], words[j], vsm = glove_dict))
        else:
            continue
    return distances


# Take a sequence and a pooling function (e.g., max, min, average)
# Calculate the semantic distances between all word pairs and pool them using the given function.
def pool_distances_within_text(text, pool = np.average, **kwarg):
    distances = distances_within_text(text, **kwarg)
    if len(distances) == 0:
        return None
    else:
        return pool(distances)
