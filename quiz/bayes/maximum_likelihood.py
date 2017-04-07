sample_memo = '''
Milt, we're gonna need to go ahead and move you downstairs into storage B. We have some new people coming in, and we need all the space we can get. So if you could just go ahead and pack up your stuff and move it down there, that would be terrific, OK?
Oh, and remember: next Friday... is Hawaiian shirt day. So, you know, if you want to, go ahead and wear a Hawaiian shirt and jeans.
Oh, oh, and I almost forgot. Ahh, I'm also gonna need you to go ahead and come in on Sunday, too...
Hello Peter, whats happening? Ummm, I'm gonna need you to go ahead and come in tomorrow. So if you could be here around 9 that would be great, mmmk... oh oh! and I almost forgot ahh, I'm also gonna need you to go ahead and come in on Sunday too, kay. We ahh lost some people this week and ah, we sorta need to play catch up.
'''

#
#   Maximum Likelihood Hypothesis
#
#
#   In this quiz we will find the maximum likelihood word based on the preceding word
#
#   Fill in the NextWordProbability procedure so that it takes in sample text and a word,
#   and returns a dictionary with keys the set of words that come after, whose values are
#   the number of times the key comes after that word.
#
#   Just use .split() to split the sample_memo text into words separated by spaces.

def NextWordProbability(sampletext,word):
    # print "NextWordProbability(",sampletext[0:20], ", ",word, ")"
    dict = {}
    word_list = sampletext.split()
    for i, x in enumerate(word_list):
        if x==word:
            word_after = word_list[i+1]
            # print i,x,word_after, dict.keys().count(word_after)
            count = 1 if (dict.keys().count(word_after) == 0) else (dict[word_after]+1)
            dict[word_after] = count

    # Can't do like below because dict is not same between inner and outer
    # dict = {word_list[i+1]: 1 if (dict.keys().count(word_list[i+1]) == 0) else (dict[word_list[i+1]]+1)
    #             for i, x in enumerate(word_list) if x==word}

    return dict

# print NextWordProbability(sample_memo,"you")
