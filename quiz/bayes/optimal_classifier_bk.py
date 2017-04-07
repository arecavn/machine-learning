#------------------------------------------------------------------

#
#   Bayes Optimal Classifier
#
#   In this quiz we will compute the optimal label for a second missing word in a row
#   based on the possible words that could be in the first blank
#
#   Finish the procedurce, LaterWords(), below
#
#   You may want to import your code from the previous programming exercise!
#
import operator

sample_memo = '''
Milt, we're gonna need to go ahead and move you downstairs into storage B. We have some new people coming in, and we need all the space we can get. So if you could just go ahead and pack up your stuff and move it down there, that would be terrific, OK?
Oh, and remember: next Friday... is Hawaiian shirt day. So, you know, if you want to, go ahead and wear a Hawaiian shirt and jeans.
Oh, oh, and I almost forgot. Ahh, I'm also gonna need you to go ahead and come in on Sunday, too...
Hello Peter, whats happening? Ummm, I'm gonna need you to go ahead and come in tomorrow. So if you could be here around 9 that would be great, mmmk... oh oh! and I almost forgot ahh, I'm also gonna need you to go ahead and come in on Sunday too, kay. We ahh lost some people this week and ah, we sorta need to play catch up.
'''

corrupted_memo = '''
Yeah, I'm gonna --- you to go ahead --- --- complain about this. Oh, and if you could --- --- and sit at the kids' table, that'd be ---
'''

data_list = sample_memo.strip().split()

words_to_guess = ["ahead","could"]

def LaterWords(sample,word,distance):
    '''@param sample: a sample of text to draw from
    @param word: a word occuring before a corrupted sequence
    @param distance: how many words later to estimate (i.e. 1 for the next word, 2 for the word after that)
    @returns: a single word which is the most likely possibility
    '''

    # TODO: Repeat the above process--for each distance beyond 1, evaluate the words that
    # might come after each word, and combine them weighting by relative probability
    # into an estimate of what might appear next.
    leaves = { word: 1.0 }
    for d in range(distance):
        print "d:", d
        next_leaves = {}

        for word, probability in leaves.items():
            relative_probs = maximum_likelihood(sample, word)
            for key, value in relative_probs.items():
                relative_probs[key] = value * probability
            print "relative_probs:", relative_probs, "(", probability, ")"

            dict_sum_update(next_leaves, relative_probs)
            # next_leaves.update(relative_probs)
            # for iword, iprobability in relative_probs.items():
            #     if iword in next_leaves: next_leaves[iword] = next_leaves[iword] + iprobability
            #     else: next_leaves[iword] = iprobability

        leaves = next_leaves
        print "leaves:", leaves

    return max(leaves.iteritems(), key=operator.itemgetter(1))[0]

def maximum_likelihood(sample, word):
    # TODO: Given a word, collect the relative probabilities of possible following words
    # from @sample. You may want to import your code from the maximum likelihood exercise.
    words = sample.split()
    next_word_count = {}
    for index in range(len(words)):
        if words[index] == word:
            next_word = words[index + 1]
            next_word_count[next_word] = next_word_count.get(next_word, 0) + 1

    total_count = sum(next_word_count.values())
    for key, value in next_word_count.items():
        next_word_count[key] = float(value) / float(total_count)

    print "next_word_count of ",word,":", next_word_count
    return next_word_count

def dict_sum_update(dict, idict):
    print "dict:", dict
    print "idict:", idict

    for iword, iprobability in idict.items():
        if iword in dict: dict[iword] = dict[iword] + iprobability
        else: dict[iword] = iprobability


    print "dict:", dict
    return dict

# sample_memo="we not pink we are pink we not pink we are black we are orange we are white we are gray we are gray"
# sample_memo="for this time for this time for this job for this time for this job for this job for this time for this time for that job for that job for that job for that job for that time for those items for those items for those items for those items for those items"
print LaterWords(sample_memo,"we",2)
