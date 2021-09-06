from lumo.contrib.string.lcs import lcs


def test_lcs():
    concat = ''.join
    assert concat(lcs('', '')) == ''
    assert concat(lcs('a', '')) == ''
    assert concat(lcs('', 'b')) == ''
    assert concat(lcs('abc', 'abc')) == 'abc'
    assert concat(lcs('abcd', 'obce')) == 'bc'
    assert concat(lcs('abc', 'ab')) == 'ab'
    assert concat(lcs('abc', 'bc')) == 'bc'
    assert concat(lcs('abcde', 'zbodf')) == 'bd'
    assert concat(lcs('aa', 'aaaa')) == 'aa'
    assert concat(lcs('GTCGTTCGGAATGCCGTTGCTCTGTAAA',
                      'ACCGGTCGAGTGCGCGGAAGCCGGCCGAA')
                  ) == 'GTCGTCGGAAGCCGGCCGAA'
