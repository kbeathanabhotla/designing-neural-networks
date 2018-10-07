from yapps import runtime

from com.designingnn.rl.CnnScanner import CnnScanner


class Cnn(runtime.Parser):
    Context = runtime.Context

    def layers(self, _parent=None):
        _context = self.Context(_parent, self._scanner, 'layers', [])
        _token = self._peek('CONV', 'NIN', 'GLOBALAVE', 'BATCHNORM', 'POOL', 'SPLIT', 'FC', 'DROP', 'SOFTMAX',
                            context=_context)
        if _token == 'CONV':
            conv = self.conv(_context)
            return conv
        elif _token == 'NIN':
            nin = self.nin(_context)
            return nin
        elif _token == 'GLOBALAVE':
            gap = self.gap(_context)
            return gap
        elif _token == 'BATCHNORM':
            bn = self.bn(_context)
            return bn
        elif _token == 'POOL':
            pool = self.pool(_context)
            return pool
        elif _token == 'SPLIT':
            split = self.split(_context)
            return split
        elif _token == 'FC':
            fc = self.fc(_context)
            return fc
        elif _token == 'DROP':
            drop = self.drop(_context)
            return drop
        else:  # == 'SOFTMAX'
            softmax = self.softmax(_context)
            return softmax

    def conv(self, _parent=None):
        _context = self.Context(_parent, self._scanner, 'conv', [])
        CONV = self._scan('CONV', context=_context)
        result = ['conv']
        numlist = self.numlist(_context)
        return result + numlist

    def nin(self, _parent=None):
        _context = self.Context(_parent, self._scanner, 'nin', [])
        NIN = self._scan('NIN', context=_context)
        result = ['nin']
        numlist = self.numlist(_context)
        return result + numlist

    def gap(self, _parent=None):
        _context = self.Context(_parent, self._scanner, 'gap', [])
        GLOBALAVE = self._scan('GLOBALAVE', context=_context)
        result = ['gap']
        numlist = self.numlist(_context)
        return result + numlist

    def bn(self, _parent=None):
        _context = self.Context(_parent, self._scanner, 'bn', [])
        BATCHNORM = self._scan('BATCHNORM', context=_context)
        return ['bn']

    def pool(self, _parent=None):
        _context = self.Context(_parent, self._scanner, 'pool', [])
        POOL = self._scan('POOL', context=_context)
        result = ['pool']
        numlist = self.numlist(_context)
        return result + numlist

    def fc(self, _parent=None):
        _context = self.Context(_parent, self._scanner, 'fc', [])
        FC = self._scan('FC', context=_context)
        result = ['fc']
        numlist = self.numlist(_context)
        return result + numlist

    def drop(self, _parent=None):
        _context = self.Context(_parent, self._scanner, 'drop', [])
        DROP = self._scan('DROP', context=_context)
        result = ['dropout']
        numlist = self.numlist(_context)
        return result + numlist

    def softmax(self, _parent=None):
        _context = self.Context(_parent, self._scanner, 'softmax', [])
        SOFTMAX = self._scan('SOFTMAX', context=_context)
        result = ['softmax']
        numlist = self.numlist(_context)
        return result + numlist

    def split(self, _parent=None):
        _context = self.Context(_parent, self._scanner, 'split', [])
        SPLIT = self._scan('SPLIT', context=_context)
        self._scan('"\\\\{"', context=_context)
        result = ['split']
        net = self.net(_context)
        result.append(net)
        while self._peek('"\\\\}"', '","', context=_context) == '","':
            self._scan('","', context=_context)
            net = self.net(_context)
            result.append(net)
        self._scan('"\\\\}"', context=_context)
        return result

    def numlist(self, _parent=None):
        _context = self.Context(_parent, self._scanner, 'numlist', [])
        self._scan('"\\\\("', context=_context)
        result = []
        NUM = self._scan('NUM', context=_context)
        result.append(int(NUM))
        while self._peek('"\\\\)"', '","', context=_context) == '","':
            self._scan('","', context=_context)
            NUM = self._scan('NUM', context=_context)
            result.append(int(NUM))
        self._scan('"\\\\)"', context=_context)
        return result

    def net(self, _parent=None):
        _context = self.Context(_parent, self._scanner, 'net', [])
        self._scan('"\\\\["', context=_context)
        result = []
        layers = self.layers(_context)
        result.append(layers)
        while self._peek('"\\\\]"', '","', context=_context) == '","':
            self._scan('","', context=_context)
            layers = self.layers(_context)
            result.append(layers)
        self._scan('"\\\\]"', context=_context)
        return result


def parse(rule, text):
    P = Cnn(CnnScanner(text))
    return runtime.wrap_error_reporter(P, rule)


if __name__ == '__main__':
    from sys import argv, stdin

    if len(argv) >= 2:
        if len(argv) >= 3:
            f = open(argv[2], 'r')
        else:
            f = stdin
        print(parse(argv[1], f.read()))
    else:
        print ('Args:  <rule> [<filename>]')
