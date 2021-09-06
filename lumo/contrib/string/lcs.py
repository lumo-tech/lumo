"""
Longest common subsequence
edited from http://wordaligned.org/articles/longest-common-subsequence
"""
import itertools
from difflib import Match, SequenceMatcher


class LCS:
    """
    A diiflib.SequenceMatch like class.

    Examples:
        a, b = 'GTCGTTCGGAATGCCGTTGCTCTGTAAA', 'ACCGGTCGAGTGCGCGGAAGCCGGCCGAA'
        l = LCS(a, b)
        print(l)
        for m in l.get_matching_blocks():
            print(m, a[m.a:m.a + m.size], b[m.b:m.b + m.size])
    """

    def __init__(self, xstr, ystr):
        self.xstr = xstr
        self.ystr = ystr

    def get_matching_blocks(self):
        """
        Return list of triples describing matching subsequences.
        """
        res = self.lcs2()
        if len(res) == 0:
            return []

        li, lj = res[0][1], res[0][3]
        size = 1
        ress = []
        for x, i, y, j in res[1:]:
            offset = (i - li) * (j - lj)
            if offset == 1:
                li, lj = i, j
                size += 1
            else:
                end = size - 1
                ress.append(Match(li - end, lj - end, size))
                li, lj = i, j
                size = 1
        end = size - 1
        ress.append(Match(li - end, lj - end, size))
        return ress

    def lcs(self):
        return lcs(self.xstr, self.ystr)

    def lcs2(self):
        return lcs2(self.xstr, self.ystr)


def lcs_lens(xs, ys):
    curr = list(itertools.repeat(0, 1 + len(ys)))
    for x in xs:
        prev = list(curr)
        for i, y in enumerate(ys):
            if x == y:
                curr[i + 1] = prev[i] + 1
            else:
                curr[i + 1] = max(curr[i], prev[i + 1])
    return curr


def lcs(xs, ys):
    """
    Return longest common subsequence list.
    """
    nx, ny = len(xs), len(ys)
    if nx == 0:
        return []
    elif nx == 1:
        return [xs[0]] if xs[0] in ys else []
    else:
        i = nx // 2
        xb, xe = xs[:i], xs[i:]
        ll_b = lcs_lens(xb, ys)
        ll_e = lcs_lens(xe[::-1], ys[::-1])
        _, k = max((ll_b[j] + ll_e[ny - j], j)
                   for j in range(ny + 1))
        yb, ye = ys[:k], ys[k:]
        return lcs(xb, yb) + lcs(xe, ye)


def lcs2(xs, ys, ii=0, kk=0):
    """
    Return longest common subsequence list with each character's index.
    """
    nx, ny = len(xs), len(ys)
    if nx == 0:
        return [['', -1, '', -1]]
    elif nx == 1:
        idx = ys.find(xs[0])
        if idx >= 0:
            return [[xs[0], ii, ys[idx], idx + kk]]
        return []
    else:
        i = nx // 2
        xb, xe = xs[:i], xs[i:]
        ll_b = lcs_lens(xb, ys)
        ll_e = lcs_lens(xe[::-1], ys[::-1])
        _, k = max((ll_b[j] + ll_e[ny - j], j)
                   for j in range(ny + 1))
        yb, ye = ys[:k], ys[k:]
        return lcs2(xb, yb, ii, kk) + lcs2(xe, ye, i + ii, k + kk)
