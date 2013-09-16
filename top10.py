## author: Mingyu Liu
## author: Shi He

def top10(s):
    """
    a helper function to find the top 10 features
    the argument should be a string of features delimited by white space
    """
    li = s.split()
    dic = {}
    for i in li:
        if i in dic:
            dic[i] += 1
        else:
            dic[i] = 1
    topCols = [(key, value) for key, value in sorted(dic.iteritems(), key = lambda(k, v) : (v, k), reverse = True)]
    topCols = topCols[: 10]
    for i in topCols:
        print i[0]
