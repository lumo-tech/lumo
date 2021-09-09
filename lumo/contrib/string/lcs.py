"""
Longest common subsequence
edited from http://wordaligned.org/articles/longest-common-subsequence
"""
import itertools
from difflib import Match, SequenceMatcher
from collections import namedtuple
from lumo.contrib.itertools import window

Chunk = namedtuple('Chunk', 'w a size')


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

    def get_matching_chunks(self):
        """
        Return list of triples describing matching subsequences.
        """
        res = self.lcs2()
        if len(res) == 0:
            return []

        li, lj = res[0][1], res[0][3]
        size = 1
        chunk_x = []
        chunk_y = []
        for x, i, y, j in res[1:]:
            offset = (i - li) * (j - lj)
            if offset == 1:
                li, lj = i, j
                size += 1
            else:
                end = size - 1
                chunk_x.append([self.xstr[li - end:li - end + size], li - end, size])
                chunk_y.append([self.ystr[lj - end:lj - end + size], lj - end, size])
                li, lj = i, j
                size = 1
        end = size - 1
        chunk_x.append([self.xstr[li - end:li - end + size], li - end, size])
        chunk_y.append([self.ystr[lj - end:lj - end + size], lj - end, size])

        # a, b = self.xstr, self.ystr

        def _merge(ress, a):
            for i in range(len(ress)):
                m = ress[i]
                l = ress[i - 1] if i > 0 else []
                r = ress[i + 1] if i < len(ress) - 1 else []
                if len(m) > 0 and m[-1] == 1:
                    if len(l) > 0 and a[l[1] + l[2]] == m[0]:
                        l[0] = l[0] + m[0]
                        l[2] += 1
                        m.clear()
                    elif len(r) > 0 and a[r[1] - 1] == m[0]:
                        r[0] = m[0] + r[0]
                        r[1] -= 1
                        r[2] += 1
                        m.clear()
            ress = [Chunk(i[0], i[1], i[2]) for i in ress if len(i) > 0]
            return ress

        chunk_x = _merge(chunk_x, self.xstr)
        chunk_y = _merge(chunk_y, self.ystr)
        return chunk_x, chunk_y

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


import numpy as np


def lcs3(text1, text2, ii=0, kk=0):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    mask = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                mask[i][j] = 0
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                mask[i][j] = 2 * int(dp[i - 1][j] > dp[i][j - 1]) - 1

    i, j = m, n
    match = []
    while i != 0 and j != 0:
        if mask[i][j] == 0:
            match.insert(0, [text1[i - 1], i - 1, text2[j - 1], j - 1])
            i, j = i - 1, j - 1
        elif mask[i][j] == 1:
            i, j = i - 1, j
        else:
            i, j = i, j - 1
    return match
