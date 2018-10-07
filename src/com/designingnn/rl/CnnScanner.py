import re

from yapps import runtime


class CnnScanner(runtime.Scanner):
    patterns = [
        ('"\\\\]"', re.compile('\\]')),
        ('"\\\\["', re.compile('\\[')),
        ('"\\\\)"', re.compile('\\)')),
        ('"\\\\("', re.compile('\\(')),
        ('"\\\\}"', re.compile('\\}')),
        ('","', re.compile(',')),
        ('"\\\\{"', re.compile('\\{')),
        ('\\s+', re.compile('\\s+')),
        ('NUM', re.compile('[0-9]+')),
        ('CONV', re.compile('C')),
        ('POOL', re.compile('P')),
        ('SPLIT', re.compile('S')),
        ('FC', re.compile('FC')),
        ('DROP', re.compile('D')),
        ('GLOBALAVE', re.compile('GAP')),
        ('NIN', re.compile('NIN')),
        ('BATCHNORM', re.compile('BN')),
        ('SOFTMAX', re.compile('SM')),
    ]

    def __init__(self, str, *args, **kw):
        runtime.Scanner.__init__(self, None, {'\\s+': None, }, str, *args, **kw)
