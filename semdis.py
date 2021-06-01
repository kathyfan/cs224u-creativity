import scipy

# takes two words and return their semantic distance in a VSM
def get_word_cosine(word1, word2, vsm):
  v1 = vsm[word1],
  v2 = vsm[word2]

  return(scipy.spatial.distance.cosine(v1, v2))