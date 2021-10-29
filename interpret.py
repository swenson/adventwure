# Copyright (c) 2016-2021 Twilio Inc.

from __future__ import print_function

import bz2
import random
import re
import sys
from collections import namedtuple


try:
    import cPickle as pickle
except ImportError:
    import pickle


global DEBUG
DEBUG = False

try:  # Python 2
    long
    raw_input
    xrange
except NameError:  # Python 3
    long = int
    raw_input = input
    xrange = range
# raw lines
Line = namedtuple('Line', 'comment label continuation statements')

# lexical analysis
Token = namedtuple('Token', 'name value')

# grammar structure
If = namedtuple('If', ['expr', 'statement'])
IfNum = namedtuple('IfNum', ['expr', 'neg', 'zero', 'pos'])
Goto = namedtuple('Goto', ['labels', 'choice'])
Assign = namedtuple('Assign', ['lhs', 'rhs'])
Comparison = namedtuple('Compare', ['a', 'op', 'b'])
Name = namedtuple('Name', ['name'])
Compound = namedtuple('Compound', ['statements'])
ArrayAssign = namedtuple('ArrayAssign', ['name', 'index', 'expr'])
Do = namedtuple('Do', ['label', 'var', 'start', 'end', 'step'])
ArrayExpr = namedtuple('ArrayExpr', ['name', 'index'])
arrayExprRegex = re.compile(r'^([a-zA-Z0-9]+)\((.*)\)$')
Call = namedtuple('Call', ['name', 'args'])
Read = namedtuple('Read', ['device', 'formatLabel', 'vars'])
Format = namedtuple('Format', ['args'])
Op = namedtuple('Op', ['op', 'a', 'b'])
Continue = ('continue',)
End = ('end',)
Stop = ('stop',)
Pause = namedtuple('Pause', ['message'])
Neg = namedtuple('Neg', ['expr'])
Type = namedtuple('Type', ['formatLabel', 'args'])
Accept = namedtuple('Accept', ['formatLabel', 'args'])
ArrayRange = namedtuple('ArrayRange', ['expr', 'var', 'start', 'end'])
Mod = namedtuple('Mod', ['num', 'mod'])
Ran = namedtuple('Ran', ['arg'])
Subroutine = namedtuple('Subroutine', ['name', 'args'])
Return = ('return',)
Dimension = namedtuple('Dimension', ['args'])
ExplDeclare = namedtuple('ExplDeclare',['type','vars'])
Implicit = namedtuple('Implicit', ['kind', 'vars'])
numericIfRegex = re.compile(r'^(\d+),(\d+),(\d+)$')

# data types
String = namedtuple('String', ['value'])
Int = namedtuple('Int', ['value'])
Float = namedtuple('Float', ['value'])

# format
IntegerFormat = namedtuple('IntegerFormat', ['count','digits'])
FloatFormat = namedtuple('FloatFormat', ['count', 'whole', 'fraction'])
AsciiFormat = namedtuple('AsciiFormat', ['count', 'read'])
asciiFormatRegex = re.compile(r'^(\d+)A(\d+)$')


commentLine = Line(True, '', False, '')

# code and data
with open('advdat.77-03-31.txt') as fin:
    data = fin.read()
# remove blank line
data = data.replace('\n\n', '\n')

with open('advf4.77-03-31.txt') as fin:
    code = fin.read()

# lexical analysis

def parse_lines(text):
    return [parse_line(line) for line in text.split('\n')]

def parse_line(line):
    comment = False
    line = line.replace('\t', ' ' * 8)
    if not line:
        return commentLine
    if line[0] in 'C*':
        return commentLine
    label = line[0:5].strip()
    if label:
        label = int(label)
    continuation = line[5] != ' '
    statements = line[6:].strip()
    if statements[0].isdigit() and statements[1] == ' ':
        continuation = True
        statements = statements[2:]
    return Line(comment, label, continuation, statements)

def combine_lines(lines):
    current_line = None
    ret = []
    for line in lines:
        if line.continuation:
            current_line = Line(False, current_line.label, False, current_line.statements + line.statements)
            continue
        elif current_line:
            ret.append(current_line)
            current_line = line
        else:
            current_line = line
    if current_line:
        ret.append(current_line)
    return ret

def with_subst(expr, name, value):
    constructor = type(expr)
    t = []
    for x in expr:
        if isinstance(x, Name) and x.name == name:
            t.append(value)
        else:
            t.append(x)
    return constructor(*t)

def tokenize(text):
    text = text.replace('.NE.', '&NE&')
    text = text.replace('.GT.', '&GT&')
    text = text.replace('.LT.', '&LT&')
    text = text.replace('.GE.', '&GE&')
    text = text.replace('.LE.', '&LE&')
    text = text.replace('.EQ.', '&EQ&')
    text = text.replace('.OR.', '&OR&')
    text = text.replace('.AND.', '&AND&')
    text = text.replace('.XOR.', '&XOR&')
    if text[0] == '"':
        # octal
        i = 1
        while i < len(text) and text[i].isdigit():
            i += 1
        return Token('NUM', int(text[1:i], 8)), i
    if text[0] == '(':
        return Token('LPAREN', '('), 1
    elif text[0] == ')':
        return Token('RPAREN', ')'), 1
    elif text[0].isalpha():
        i = 0
        while i < len(text) and (text[i].isalpha() or text[i].isdigit()):
            i += 1
        return Token('WORD', text[:i]), i
    elif text.startswith(('&NE&', '&GT&', '&LT&', '&GE&', '&LE&', '&EQ&', '&OR&')):
        return Token('OP', '.' + text[1:3] + '.'), 4
    elif text.startswith('&AND&') or text.startswith('&XOR&'):
        return Token('OP', '.' + text[1:4] + '.'), 5
    elif text[0].isdigit():
        i = 0
        while i < len(text) and (text[i].isdigit() or text[i] == '.'):
            i += 1
        return Token('NUM', text[:i]), i
    elif text[0] == '-' and len(text) > 1 and text[1].isdigit():
        i = 1
        while i < len(text) and (text[i].isdigit() or text[i] == '.'):
            i += 1
        return Token('NUM', text[:i]), i
    elif text[0] == "'":
        r = text.index("'", 1)
        return Token('STRING', text[1:r]), r + 1
    elif text[0] in '+-*/=':
        return Token('OP', text[0]), 1
    elif text[0] == ',':
        return Token('COMMA', text[0]), 1
    print('tokenize', text)
    exit()

def text_to_tokens(text):
    if isinstance(text, Token):
        if text.name == 'WORD':
            return Name(text.value)
        if text.name == 'NUM':
            return Int(int(text.value))
        if text.name == 'STRING':
            return String(text.value)
        if text.name == 'EXPR':
            return text.value
        print('unknown token type', text)
        exit()
    elif isinstance(text, list):
        tokens = text
    else:
        tokens = []
        c = 0
        while c < len(text):
            if text[c] == ' ' or text[c] == '\t':
                c += 1
                continue
            t, skip = tokenize(text[c:])
            c += skip
            tokens.append(t)
    return tokens

def parenthesize_arrays(tokens, start_after=0):
    lparens = [i for i, t in enumerate(tokens) if t.name == 'LPAREN' and i > start_after]
    new_lparens = []
    for i in lparens:
        if tokens[i - 1].name != 'WORD':
            continue
        new_lparens.append(i)
    lparens = new_lparens

    if not lparens:
        return tokens

    l = lparens[0]
    r = match_right_paren_tokens(tokens[l:]) + l
    expr_list = parse_expr_list(tokens[l+1:r])
    if len(expr_list) == 1:
        expr_list = expr_list[0]
    expr = Token('EXPR', expr_list)
    # add back in parens
    add = [Token('LPAREN', '('), expr, Token('RPAREN', ')')]
    new_tokens = tokens[:l] + add + tokens[r+1:]
    return parenthesize_arrays(new_tokens, l+1)

def parenthesize(tokens):
    lparens = [i for i, t in enumerate(tokens) if t.name == 'LPAREN']
    new_lparens = []
    for i in lparens:
        if i == 0:
            new_lparens.append(i)
            continue
        if tokens[i - 1].name == 'WORD':
            continue
        new_lparens.append(i)
    lparens = new_lparens

    if not lparens:
        return tokens

    l = lparens[0]
    r = match_right_paren_tokens(tokens[l:]) + l
    expr = Token('EXPR', parse_expr(tokens[l+1:r]))
    new_tokens = tokens[:l] + [expr] + tokens[r+1:]
    return parenthesize(new_tokens)

def split_parse_bigops(tokens, op=None):
    if op is None:
        tokens = split_parse_bigops(tokens, '.OR.')
        tokens = split_parse_bigops(tokens, '.XOR.')
        tokens = split_parse_bigops(tokens, '.AND.')
        tokens = split_parse_bigops(tokens, '.EQ.')
        tokens = split_parse_bigops(tokens, '.NE.')
        tokens = split_parse_bigops(tokens, '.GT.')
        tokens = split_parse_bigops(tokens, '.GE.')
        tokens = split_parse_bigops(tokens, '.LT.')
        tokens = split_parse_bigops(tokens, '.LE.')
        tokens = split_parse_bigops(tokens, '+')
        tokens = split_parse_bigops(tokens, '-')
        return tokens
    ors = [i for i, t in enumerate(tokens) if t.name == 'OP' and t.value == op]
    if not ors:
        return tokens
    o = ors[0]
    if o == 0 and op == '-':
        a = Int(0)
    else:
        a = parse_expr(tokens[:o])
    b = parse_expr(tokens[o+1:])
    return [Token('EXPR', Op(op, a, b))]

def parse_expr(text):
    tokens = text_to_tokens(text)
    tokens = parenthesize_arrays(tokens)
    tokens = parenthesize(tokens)
    tokens = split_parse_bigops(tokens)
    return parse_expr_piece(tokens)[0]

def parse_expr_piece(tokens):
    # expr -> ( expr )
    # expr -> NAME ( expr )
    # expr -> expr .OR. expr
    # subexpr -> expr OP expr
    # expr -> WORD | NUM | STRING
    if not tokens:
        return None, 0
    if isinstance(tokens, Token):
        tokens = [tokens]
    # print 'parse', tokens
    if len(tokens) == 1:
        if tokens[0].name == 'WORD':
            return Name(tokens[0].value), 1
        if tokens[0].name == 'NUM' and '.' in str(tokens[0].value):
            return Float(float(tokens[0].value)), 1
        if tokens[0].name == 'NUM':
            return Int(int(tokens[0].value)), 1
        if tokens[0].name == 'STRING':
            return String(tokens[0].value), 1
        if tokens[0].name == 'EXPR':
            return tokens[0].value, 1
        print('unknown token type', tokens[0])
        exit()
    if len(tokens) >= 4 and tokens[0].name == 'WORD' and tokens[1].name == 'LPAREN':
        name = parse_expr_piece(tokens[:1])[0]
        r = match_right_paren_tokens(tokens)
        expr = parse_expr_list(tokens[2:r])
        if r == len(tokens) - 1:
            return ArrayExpr(name, expr), len(tokens)
        t = Token('EXPR', ArrayExpr(name, expr))
        new_tokens = [t] + tokens[r+1:]
        result, skip = parse_expr_piece(new_tokens)
        return result, skip + r + 1

    if tokens[0].name == 'LPAREN':
        r = match_right_paren_tokens(tokens)
        expr = parse_expr_piece(tokens[1:r])[0]
        if r == len(tokens) - 1:
            return expr, len(tokens)
        t = Token('EXPR', expr)
        new_tokens = [t] + tokens[r+1:]
        result, skip = parse_expr_piece(new_tokens)
        return result, skip - 1 + r + 1

    if tokens[1].name == 'OP':
        second, skip = parse_expr_piece(tokens[2:])
        op = Op(tokens[1].value, parse_expr_piece([tokens[0]])[0], second)
        opExpr = Token('EXPR', op)
        new_tokens = [opExpr] + tokens[2+skip:]
        if len(new_tokens) == 1:
            return op, len(tokens)
        return parse_expr_piece(new_tokens)
    # special case subtraction
    if len(tokens) >= 2 and tokens[0].name == 'WORD' and tokens[1].name == 'NUM' and str(tokens[1].value).startswith('-'):
        return parse_expr_piece([tokens[0], Token('OP', '-'),
            Token(tokens[1].name, str(tokens[1].value)[1:])] + tokens[2:])
    # implicit 0
    if tokens[0].name == 'OP' and tokens[0].value == '-':
        return parse_expr_piece([Token('NUM', '0')] + tokens)
    print('unknown expr', tokens)
    exit()

def parse_expr_list(text):
    tokens = text_to_tokens(text)
    if not tokens:
        return ()
    if tokens[0].name == 'COMMA':
        tokens = tokens[1:]
    if len(tokens) == 1:
        return (parse_expr(tokens),)
    if not any(t.name == 'COMMA' for t in tokens):
        return (parse_expr(tokens),)
    if tokens[1].name == 'COMMA':
        return (parse_expr(tokens[:1]),) + parse_expr_list(tokens[1:])
    return tuple([parse_expr(x) for x in smart_split_tokens(tokens)])
    print('parse list', tokens)
    exit()

def parse_expr_list2(text):
    if not text.strip():
        return ()
    if text.startswith(','):
        text = text[1:]
    if text.startswith("'"):
        text = text[1:]
        end = text.index("'")
        return (String(text[:end]),) + parse_expr_list2(text[end+1:])
    if ',' not in text:
        return (parse_expr(text),)
    if '(' not in text:
        return tuple([parse_expr(x) for x in text.split(',')])
    t = tuple([parse_expr(x) for x in smart_split(text)])
    print('parse list', text, t)
    exit()

# paren-aware comma-split
def smart_split(text):
    parts = []
    current = ''
    level = 0
    for c in text:
        if c == ',':
            if level == 0:
                parts.append(current)
                current = ''
                continue
        if c == '(':
            level += 1
        if c == ')':
            level -= 1
        current += c
    if text[-1] == ',':
        parts.append('')
    if current:
        parts.append(current)
    return parts

# paren-aware comma-split
def smart_split_tokens(tokens):
    parts = []
    current = []
    level = 0
    for c in tokens:
        if c.name == 'COMMA':
            if level == 0:
                parts.append(current)
                current = []
                continue
        if c.name == 'LPAREN':
            level += 1
        if c.name == 'RPAREN':
            level -= 1
        current.append(c)
    if tokens[-1].name == 'COMMA':
        parts.append([])
    if current:
        parts.append(current)
    return parts

def parse_format_item(text):
    if text == 'G':
        return FloatFormat(0, 15, 7)
    if text.endswith('G'):
        return FloatFormat(int(text[:-1]), 15, 7)
    if asciiFormatRegex.match(text):
        m = asciiFormatRegex.match(text)
        return AsciiFormat(int(m.group(1)), m.group(2))
    if text.startswith('A'):
        return AsciiFormat(1, int(text[1:]))
    if text.startswith('I'):
        return IntegerFormat(1,int(text[1:]))
    if text.startswith("'"):
        return String(text[1:-1])
    if text == '/':
        return None
    print('format', text)
    exit()

def parse_format_list(text):
    return tuple([parse_format_item(x) for x in text.split(',')])

def match_right_paren(text, which=0):
    current = 0
    for i, c in enumerate(text):
        if c == '(':
            current += 1
        if c == ')':
            current -= 1
            if current == which:
                return i
    return -1
def match_right_paren_tokens(tokens):
    current = 0
    for i, c in enumerate(tokens):
        if c.name == 'LPAREN':
            current += 1
        if c.name == 'RPAREN':
            current -= 1
            if current == 0:
                return i
    return -1

def with_set(args, name, value):
    new_args = []
    # hack
    if len(args) == 1 and isinstance(args[0], tuple) and not hasattr(args[0], '__slots__'):
        args = list(args[0])
    for arg in args:
        if isinstance(arg, Name) and arg.name == name:
            new_args.append(value)
        else:
            new_args.append(arg)
    return tuple(new_args)

def parse_type_list(text):
    text = text.strip()
    if text[0] == '(':
        # this is an array expression
        text = text[1:-1]
        parts = smart_split(text)
        var, start = parts[1].split('=')
        var = var.strip()
        start = parse_expr(start)
        end = parse_expr(parts[2])
        arr = parse_expr_list(parts[0])
        return ArrayRange(arr, var, start, end)
    else:
        return parse_expr_list2(text)

# state
class Game(object):
    def getstate(self):
        d = dict(
            globals=self.globals,
            globImplicit=self.globImplicit,
            subroutines=self.subroutines,
            substack=self.substack,
            stmtstack=self.stmtstack,
            current=self.current,
            varstack=self.varstack,
            progstack=self.progstack,
            dostack=self.dostack,
            prog=self.prog,
            labels=self.labels,
            current_subroutine=self.current_subroutine,
            waiting_for_user=self.waiting_for_user)
        return bz2.compress(pickle.dumps(d))

    def setstate(self, incoming):
        data = pickle.loads(bz2.decompress(incoming))
        self.globals = data['globals']
        self.globImplicit = data['globImplicit']
        self.subroutines = data['subroutines']
        self.substack = data['substack']
        self.stmtstack = data['stmtstack']
        self.current = data['current']
        self.varstack = data['varstack']
        self.progstack = data['progstack']
        self.dostack = data['dostack']
        self.prog = data['prog']
        self.labels = data['labels']
        self.current_subroutine = data['current_subroutine']
        self.waiting_for_user = data['waiting_for_user']

    def waiting(self):
        return self.waiting_for_user

    def __init__(self, handler):
        self.handler = handler
        self.globals = {}
        self.globImplicit = {}
        self.subroutines = {}
        self.current_subroutine = '__main__'
        self.substack = ['__main__']
        self.stmtstack = []
        self.current = 0
        self.varstack = [self.globals]
        self.progstack = []
        self.dostack = []
        self.data_cursor = 0
        self.prog = []
        self.labels = {'__main__': {}}
        self.words = None
        self.data = data
        self.waiting_for_user = False

        lines = combine_lines(parse_lines(code))
        for line in lines:
            if line.comment:
                continue
            self.words = line.statements.split(' ')

            if self.words[0] == 'IMPLICIT' and self.current_subroutine == '__main__':
                kind = 'REAL'
                if self.words[1].startswith('INTEGER'):
                    kind = 'INTEGER'
                pieces = self.words[1].split('(')
                names = pieces[1][:-1].split('-')
                vars = []
                for c in xrange(ord(names[0]), ord(names[1]) + 1):
                    vars.append(chr(c))
                if kind == 'INTEGER':
                    self.globImplicit['IMPLICIT-INT'] = Implicit(kind, vars)
                else:
                    self.globImplicit['IMPLICIT-REAL'] = Implicit(kind. vars)
            if self.words[0] == 'REAL':
                pass
            if self.words[0] == 'COMMON':
                continue
            if self.words[0] == 'INTEGER':
                pass
            if self.words[0] == 'DIMENSION' and self.current_subroutine == '__main__':
                # array of numbers
                pieces = self.words[1].split('),')
                for piece in pieces:
                    name = piece.split('(')[0]
                    size = piece.split('(')[1].rstrip(')')
                    if ',' in size:
                        a, b = size.split(',')
                        a = int(a)
                        b = int(b)
                        size = (a, b)
                    else:
                        size = int(size)

                    # If Implicit statement exists
                    # specifing first letter of variable use Implicit to type variable
                    defaultVal = 0.0
                    if self.globImplicit.get('IMPLICIT-INT',None) != None:
                        if name[0] in self.globImplicit['IMPLICIT-INT'].vars:
                            defaultVal = 0
                    elif self.globImplicit.get('IMPLICIT-REAL',None) != None:
                        if varname[0] in self.globImplicit['IMPLICIT-REAL'].vars:
                            defaultVal = 0.0

                    self.globals[name] = make_dimension(size,defaultVal)
                continue
            statement = self.parse_statement(line.statements)
            if statement is None:
                continue
            if isinstance(statement, Subroutine):
                self.current_subroutine = statement.name
                self.labels[self.current_subroutine] = {}
            if DEBUG:
                print()
                print(line.statements)
                print(statement)
                print()
            self.prog.append(statement)
            if line.label:
                self.labels[self.current_subroutine][line.label] = len(self.prog) - 1
                if DEBUG:
                    print(line.label)
            if isinstance(statement, Subroutine):
                self.subroutines[statement.name] = len(self.prog) - 1

    def parse_statement(self, statement):
        if statement.startswith('IF ') or statement.startswith('IF('):
            # parse if-statement
            statement = statement[2:].strip()
            r = match_right_paren(statement)
            expr = parse_expr(statement[1:r].strip())
            stmt = statement[r+1:].strip()
            if numericIfRegex.match(stmt):
                # numerical if
                m = numericIfRegex.match(stmt)
                a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
                return IfNum(expr, a, b, c)
            stmt = self.parse_statement(stmt)
            return If(expr, stmt)
        elif statement.startswith('GOTO') or statement.startswith('GO TO'):
            if statement.startswith('GO TO'):
                statement = statement[5:].strip()
            else:
                statement = statement[4:].strip()
            if statement.startswith('('):
                # list
                j = statement.index(')')
                gotoLabels = tuple([int(x) for x in statement[1:j].split(',')])
                choice = parse_expr(statement[j+1:])
                return Goto(gotoLabels, choice)
            else:
                return Goto((int(statement),), Int(1))
        elif statement.startswith('DATA'):
            lhs, rhs, _ = statement[4:].split('/')
            if lhs.startswith('('):
                i = lhs.index(',')
                name, rang = lhs[1:i], lhs[i+1:-1]
                name = name[:name.index('(')].strip()
                rang = int(rang[2:rang.index(',')]), int(rang[rang.index(',')+1:])
                length = rang[1] - rang[0] + 1
                statements = []
                rhs = rhs.split(',')
                for i in xrange(rang[0], rang[1] + 1):
                    statements.append(ArrayAssign(name, i, parse_expr(rhs[i - 1])))
                return Compound(statements)
            else:
                name = lhs.strip()
                rhs = rhs.split(',')
                statements = []
                for i in xrange(len(rhs)):
                    statements.append(ArrayAssign(name, i + 1, parse_expr(rhs[i])))
                return Compound(statements)
            print('data', lhs, rhs)
            exit()
        elif statement.startswith('DO'):
            statement = statement[2:].strip()
            i = statement.index(' ')
            label = int(statement[:i])
            var, rang = statement[i+1:].split('=')
            rang = rang.split(',')
            start = parse_expr(rang[0])
            end = parse_expr(rang[1])
            step = 1
            if len(rang) == 3:
                step = int(rang[2])
            return Do(label, var, start, end, step)
        elif statement.startswith('CALL'):
            statement = statement[4:].strip()
            i = statement.index('(')
            name = statement[:i]
            args = statement[i+1:-1]
            args = [parse_expr(arg) for arg in args.split(',')]
            return Call(name, args)
        elif statement.startswith('READ'):
            statement = statement[4:].strip()
            i = statement.index('(')
            j = statement.index(')')
            a, b = statement[i+1:j].split(',')
            source = int(a)
            formatLabel = int(b)
            args = statement[j+1:].strip()
            # hack hack hack
            args = args.split(',')
            if len(args) == 1:
                return Read(source, formatLabel, (parse_expr(args[0]),))
            elif len(args) == 2:
                a, b = parse_expr(args[0]), parse_expr(args[1])
                return Read(source, formatLabel, (a, b))
            else:
                # hack
                a = parse_expr(args[0])
                read_args = [a]
                if args[1].startswith('('):
                    b = ','.join(args[1:])
                    expr = b[1:-1]
                    l, r = expr.split(')')
                    i = l.index('(')
                    var = l[:i]
                    largs = [parse_expr(x) for x in l[i+1:].split(',')]
                    r = r[1:]
                    rname, rrang = r.split('=')
                    ra, rb = rrang.split(',')
                    ra = int(ra)
                    rb = int(rb)
                    # hack
                    for i in xrange(ra, rb + 1, 1):
                        read_args.append(ArrayExpr(var, with_set(largs, rname, Int(i))))
                    return Read(source, formatLabel, read_args)
                # hack
                read_args.append(parse_expr(args[1]))
                b = ','.join(args[2:])
                expr = b[1:-1]
                l, r = expr.split(')')
                i = l.index('(')
                var = l[:i]
                largs = [parse_expr(x) for x in l[i+1:].split(',')]
                r = r[1:]
                rname, rrang = r.split('=')
                ra, rb = rrang.split(',')
                ra = int(ra)
                rb = int(rb)
                # hack
                for i in xrange(ra, rb + 1, 1):
                    read_args.append(ArrayExpr(var, with_set(largs, rname, Int(i))))
                return Read(source, formatLabel, read_args)
            print('read', args)
            exit()
        elif statement.startswith('FORMAT'):
            return Format(parse_format_list(statement[7:-1]))
        elif statement.strip() == 'CONTINUE':
            return Continue
        elif statement.strip() == 'STOP':
            return Stop
        elif statement.startswith('PAUSE'):
            return Pause(parse_expr(statement[5:].strip()))
        elif statement.startswith('TYPE'):
            statement = statement[4:].strip()
            if ',' in statement:
                i = statement.index(',')
                ll = parse_type_list(statement[i+1:])
                return Type(int(statement[:i]), ll)
            else:
                return Type(int(statement), ())
        elif statement.startswith('ACCEPT'):
            statement = statement[6:].strip()
            if ',' in statement:
                i = statement.index(',')
                return Accept(int(statement[:i]), parse_type_list(statement[i+1:]))
            else:
                return Accept(int(statement), ())
        elif statement.strip() == 'END':
            return End
        elif statement.startswith('SUBROUTINE'):
            statement = statement[10:].strip()
            i = statement.index('(')
            name = statement[:i].strip()
            args = parse_expr_list2(statement[i+1:-1])
            return Subroutine(name, args)
        elif statement.startswith('RETURN'):
            return Return
        elif statement.startswith('IMPLICIT'):
            text = statement[9:].strip()
            kind = 'REAL'
            if text.startswith('INTEGER'):
                kind = 'INTEGER'
            else:
                print('halt on', stmt)
                exit()
            pieces = text.split('(')
            names = pieces[1][:-1].split('-')
            vars = []
            for c in xrange(ord(names[0]), ord(names[1]) + 1):
                vars.append(chr(c))
            return Implicit(kind, vars)
        elif statement.startswith('DIMENSION'):
            pieces = self.words[1].split('),')
            args = []
            for piece in pieces:
                name = piece.split('(')[0]
                size = piece.split('(')[1].rstrip(')')
                if ',' in size:
                    a, b = size.split(',')
                    a = int(a)
                    b = int(b)
                    args.append((name, (a, b)))
                else:
                    size = int(size)
                    args.append((name, size))
            return Dimension(tuple(args))
        elif statement.startswith('INTEGER'):
            pieces = self.words[1].split(',')
            args = []
            for piece in pieces:
                args.append(piece)
            return ExplDeclare('INTEGER',args)
        elif statement.startswith('REAL'):
            pieces = self.words[1].split(',')
            args = []
            for piece in pieces:
                args.append(piece)
            return ExplDeclare('REAL',args)
        if '=' in statement:
            lhs, rhs = statement.split('=')
            lhs = parse_expr(lhs)
            rhs = parse_expr(rhs)
            return Assign(lhs, rhs)
        print('statement', statement)
        exit()

    def go(self):
        self.current_subroutine = '__main__'
        while True:
            self.current = self.execute(self.current)

    def read_float(self, format):
        while self.data_cursor < len(self.data) and (self.data[self.data_cursor] == ' ' or self.data[self.data_cursor] == '\t'):
            self.data_cursor += 1
        x = ''
        while self.data_cursor < len(self.data) and (self.data[self.data_cursor].isdigit() or self.data[self.data_cursor] == '.' or self.data[self.data_cursor] == '-'):
            x += self.data[self.data_cursor]
            self.data_cursor += 1
        if x == '':
          return 0.0
        return float(x)

    def read_chars(self, num):
        # skip tabs
        while self.data_cursor < len(self.data) and self.data[self.data_cursor] == '\t':
            self.data_cursor += 1
        x = ''
        while len(x) < 5 and self.data_cursor < len(self.data) and self.data[self.data_cursor] != '\r' and self.data[self.data_cursor] != '\n':
            x += self.data[self.data_cursor]
            self.data_cursor += 1
        return x

    def next_record(self):
        if self.data_cursor >= len(self.data):
            return
        self.data_cursor = self.data.index('\n', self.data_cursor) + 1

    def execute_read(self, format, vars):
        ai = 0
        vi = 0
        while ai < len(format.args) and vi < len(vars):
            arg = format.args[ai]
            ai += 1
            if isinstance(arg, FloatFormat):
                count = 1
                if arg.count > 0:
                    count = arg.count
                for c in xrange(count):
                    var = vars[vi]
                    vi += 1
                    f = self.read_float(arg)
                    self.assign(var, f)
                continue
            elif isinstance(arg, AsciiFormat):
                for c in xrange(arg.count):
                    var = vars[vi]
                    vi += 1
                    chars = self.read_chars(int(arg.read))
                    self.assign(var, chars)
                continue
            elif isinstance(arg, IntegerFormat):
                count = 1
                if arg.count > 0:
                    count = arg.count
                for c in xrange(count):
                    var = vars[vi]
                    vi += 1
                    f = int(self.read_float(arg))
                    self.assign(var, f)
                continue
            print('halt on format', format, vars)
            exit()
        self.next_record()

    def lookup(self, varname):
        if isinstance(varname, Name):
            return self.lookup(varname.name)
        for vars in self.varstack[::-1]:
            if varname in vars:
                return vars[varname]
        if varname == 'SETUP':
            return 0
        return 0

    def assign(self, varname, value):
        if isinstance(varname, Name):
            return self.assign(varname.name, value)
        if isinstance(varname, ArrayExpr):
            self.array_assign(varname.name, self.eval_expr(varname.index), value)
            return
        #for i in xrange(len(self.varstack) - 1, -1, -1):
        # if variable instance already exists do not change the type of variable stored
        if varname in self.varstack[-1]:
            if isinstance(self.varstack[-1][varname],int) and isinstance(value,float):
                self.varstack[-1][varname] = int(value)
            elif isinstance(self.varstack[-1][varname],float) and isinstance(value,int):
                self.varstack[-1][varname] = float(value)
            else:
                self.varstack[-1][varname] = value
            return

        # If variable instance doesn't exist yet and Implicit statement exists
        # specifing first letter of variable use Implicit to type variable
        if self.varstack[-1].get('IMPLICIT-INT',None) != None:
            if varname[0] in self.varstack[-1]['IMPLICIT-INT'].vars and isinstance(value,float):
                self.varstack[-1][varname] = int(value)
                return
        if self.varstack[-1].get('IMPLICIT-REAL',None) != None:
            if varname[0] in self.varstack[-1]['IMPLICIT-REAL'].vars and isinstance(value,int):
                self.varstack[-1][varname] = float(value)
                return

        self.varstack[-1][varname] = value

    def array_assign(self, varname, index, value):
        if isinstance(varname, Name):
            varname = varname.name
        if isinstance(index, tuple):
            if len(index) == 1:
                index = index[0]
        #for i in xrange(len(self.varstack) - 1, -1, -1):
        for i in xrange(-1,-2,-1):
            if varname in self.varstack[i]:
                v = self.varstack[i][varname][2]
                if isinstance(index, tuple):
                    for x in index[:-1]:
                        v = v[int(x) - 1]
                    v[int(index[-1]) - 1] = value
                    return
                v[int(index) - 1] = value
                return
        print('failed array lookup', varname, [list(v) for v in self.varstack])
        exit()

    def eval_expr(self, expr):
        result = self.eval_expr_inner(expr)
        if DEBUG:
            print('eval expr', expr, '->', result)
        return result

    def eval_expr_inner(self, expr):
        if isinstance(expr, int):
            return expr
        if isinstance(expr, str):
            return expr
        if isinstance(expr, Op):
            a = self.eval_expr(expr.a)
            b = self.eval_expr(expr.b)
            if isinstance(a, str):
                a = string_to_dec_num(a)
            if isinstance(b, str):
                b = string_to_dec_num(b)
            if expr.op == '.XOR.':
                return a ^ b
            if expr.op == '.AND.':
                a = int(a)
                b = int(b)
                return a & b
            if expr.op == '.OR.':
                return a | b
            if expr.op == '.NE.':
                return (not equals(a, b)) * -1
            if expr.op == '.EQ.':
                return equals(a, b) * -1
            if expr.op == '.LE.':
                return (a <= b) * -1
            if expr.op == '.LT.':
                return (a < b) * -1
            if expr.op == '.GT.':
                return (a > b) * -1
            if expr.op == '.GE.':
                return (a >= b) * -1
            if expr.op == '+':
                return wrap(a + b)
            if expr.op == '-':
                return wrap(a - b)
            if expr.op == '*':
                return wrap(a * b)
            if expr.op == '/':
                # hack
                return int(a) / int(b)
        if isinstance(expr, Name):
            l = self.lookup(expr.name)
            return l
        if isinstance(expr, Int):
            return expr.value
        if isinstance(expr, Float):
            return expr.value
        if isinstance(expr, ArrayExpr):
            try:
                if expr.name.name == 'RAN':
                    # magic function
                    return random.random()
                idx = self.eval_expr(expr.index)
                if expr.name.name == 'MOD':
                    # magic function
                    return idx[0] % idx[1]
            except AttributeError:
                pass
            idx = self.eval_expr(expr.index)
            raw_arr = self.lookup(expr.name)
            if not isinstance(raw_arr, list):
                exit("Array variable %s is not array in %s" % (expr.name, expr))
            arr = self.lookup(expr.name)[2]
            if isinstance(idx, tuple):
                for x in idx[:-1]:
                    arr = arr[int(x) - 1]
                return arr[int(idx[-1]) - 1]
            else:
                return arr[int(idx) - 1]
        if isinstance(expr, tuple) and len(expr) == 1:
            return self.eval_expr(expr[0])
        if isinstance(expr, tuple) and not hasattr(expr, '__slots__'):
            return tuple([self.eval_expr(x) for x in expr])
        print('halt on eval expr', expr)
        exit()

    # output
    def execute_type(self, format, vars):
        # if format.args[0] is None and not vars:
        #     sys.stdout.flush()

        if isinstance(vars, ArrayRange):
            # hack
            expr = vars.expr
            if isinstance(expr, tuple) and not hasattr(expr, '__slot__') \
                and len(expr) == 1:
                expr = expr[0]
            name = expr.name
            if isinstance(name, Name):
                name = name.name
            var = vars.var
            if isinstance(var, Name):
                var = var.name
            start = self.eval_expr(vars.start)
            end = self.eval_expr(vars.end)
            new_vars = []
            for i in xrange(start, start + end):
                args = with_set(expr.index, var, Int(i))
                new_vars.append(ArrayExpr(name, args))
            vars = new_vars
        ai = 0
        vi = 0
        while ai < len(format.args) and vi < len(vars):
            arg = format.args[ai]
            ai += 1
            if isinstance(arg, AsciiFormat):
                for c in xrange(arg.count):
                    if vi >= len(vars):
                        break
                    var = vars[vi]
                    vi += 1

                    self.handler.write(to_string(self.eval_expr(var)))
                continue
            elif isinstance(arg, IntegerFormat) or isinstance(arg, FloatFormat):
                for c in xrange(arg.count):
                    if vi >= len(vars):
                        break
                    var = vars[vi]
                    vi += 1

                    self.handler.write(str(self.eval_expr(var)))
                continue
            elif isinstance(arg, String):
                self.handler.write(arg.value)
                continue
            print('halt on format', format, vars)
            exit()
        self.handler.write("\n")

    def execute_accept(self, format, vars):
        global DEBUG
        if isinstance(vars, ArrayRange):
            # hack
            if not hasattr(vars.expr, 'name'):
                name = vars.expr[0].name.name
                index = vars.expr[0].index[0].name
            else:
                name = vars.expr.name.name
                index = vars.expr.index[0].name
            new_vars = []
            for i in xrange(self.eval_expr(vars.start), self.eval_expr(vars.end) + 1):
                new_vars.append(ArrayExpr(name, Int(i)))
            vars = new_vars
        self.waiting_for_user = True
        line = self.handler.read()
        if line == "*":
            DEBUG = not DEBUG
            line = ""
        self.waiting_for_user = False
        old_data = self.data
        old_data_cursor = self.data_cursor
        self.data = line
        self.data_cursor = 0
        try:
            ai = 0
            vi = 0
            while ai < len(format.args) and vi < len(vars):
                arg = format.args[ai]
                ai += 1
                if isinstance(arg, AsciiFormat):
                    for c in xrange(arg.count):
                        var = vars[vi]
                        vi += 1
                        chars = self.read_chars(int(arg.read)).upper()
                        self.assign(var, chars)
                    continue
        finally:
            self.data = old_data
            self.data_cursor = old_data_cursor

    def execute(self, current):
        next = self.execute_statement(self.prog[current], current)
        if next is None:
            next = self.current + 1
        if next == -1 or \
            (self.dostack and self.dostack[-1][1] == self.current and next == self.current + 1):
            # return to the beginning of the Do
            return self.dostack[-1][0]
        return next

    def execute_statement(self, stmt, current):
        if DEBUG:
            print('execute', stmt)
        if isinstance(stmt, If):
            expr = self.eval_expr(stmt.expr)
            if isinstance(expr, bool) or isinstance(expr, int):
                if expr:
                    return self.execute_statement(stmt.statement, current)
                else:
                    return
        elif isinstance(stmt, IfNum):
            expr = self.eval_expr(stmt.expr)
            if expr < 0:
                return self.labels[self.current_subroutine][stmt.neg]
            elif expr == 0:
                return self.labels[self.current_subroutine][stmt.zero]
            else:
                return self.labels[self.current_subroutine][stmt.pos]
        elif isinstance(stmt, Assign):
            if isinstance(stmt.lhs, Name):
                val = self.eval_expr(stmt.rhs)
                self.assign(stmt.lhs.name, val)
                return
            elif isinstance(stmt.lhs, ArrayExpr):
                self.array_assign(stmt.lhs.name.name, self.eval_expr(stmt.lhs.index), self.eval_expr(stmt.rhs))
                return
        elif isinstance(stmt, ArrayAssign):
            val = self.eval_expr(stmt.expr)
            self.array_assign(stmt.name, self.eval_expr(stmt.index), val)
            return
        elif isinstance(stmt, Compound):
            for s in stmt.statements:
                self.execute_statement(s, current)
            return
        elif isinstance(stmt, Do):
            if self.dostack and self.current == self.dostack[-1][0]:
                # this is the currently executing Do
                _, final, i, start, end, step = self.dostack[-1]
                if i == end - 1:
                    return final + 1
                else:
                    i = i + step
                self.assign(stmt.var, i)
                self.dostack[-1][2] = i
                return
            else:
                # this is a new Do
                start = self.eval_expr(stmt.start)
                end = self.eval_expr(stmt.end) + 1
                step = self.eval_expr(stmt.step)
                self.dostack.append([self.current, self.labels[self.current_subroutine][stmt.label], start, start, end, step])
                self.assign(stmt.var, start)
                return

        elif isinstance(stmt, Call):
            if stmt.name == 'IFILE':
                # ignore setting input
                return
            self.stmtstack.append(stmt)
            subr_start = self.subroutines[stmt.name]
            # create a new frame
            sub = self.prog[subr_start]
            self.varstack.append({})
            for var, val in zip(sub.args, stmt.args):
                self.varstack[-1][var.name] = self.eval_expr(val)
            self.progstack.append(current)
            self.current_subroutine = stmt.name
            self.substack.append(self.current_subroutine)
            return subr_start + 1
        elif stmt == Return:
            # populate return variables
            callstmt = self.stmtstack.pop()
            subr_start = self.subroutines[callstmt.name]
            sub = self.prog[subr_start]
            assigns = {}
            for var, val in zip(sub.args, callstmt.args):
                if isinstance(val, Name):
                    # lookup the value and return it
                    assigns[val.name] = (var, self.lookup(var))
            self.substack.pop()
            self.current_subroutine = self.substack[-1]
            self.varstack.pop()
            for var, (original_var, val) in assigns.items():
                if DEBUG:
                    print('return %s variable %s -> %s -> %s' % (callstmt.name, var, original_var, val))
                self.assign(var, val)
            return self.progstack.pop() + 1
        elif isinstance(stmt, Read):
            if stmt.device == 1:
                format = self.prog[self.labels[self.current_subroutine][stmt.formatLabel]]
                self.execute_read(format, stmt.vars)
                return
        elif isinstance(stmt, Format):
            return
        elif stmt == End:
            if self.current_subroutine == '__main__':
                print('Program ended')
                exit()
            return
        elif isinstance(stmt, Goto):
            choice = int(self.eval_expr(stmt.choice)) - 1
            if DEBUG:
                print('goto choice', choice)
            if choice < 0 or choice >= len(stmt.labels):
                # just ignore
                return
            g = self.labels[self.current_subroutine][stmt.labels[choice]]
            # did we break out of the last Do?
            if self.dostack:
                start = self.dostack[-1][0]
                end = self.dostack[-1][1]
                if g < start or g > end:
                    self.dostack.pop()
            return g
        elif stmt == Continue:
            _, final, i, start, end, step = self.dostack[-1]
            if i == end - 1:
                # we are done with this Do... pop it off
                self.dostack.pop()
                # ... but we might have another Do
                if self.dostack and self.dostack[-1][1] == self.current:
                    return self.dostack[-1][0]
                # otherwise we just go on our merry way
                return final + 1
            else:
                # return to the beginning of the Do
                return self.dostack[-1][0]
            return -1
        elif isinstance(stmt, Pause):
            self.handler.write(stmt.message.value + "\n")
            if stmt.message != 'INIT DONE':
                return
            # wait for user input
            self.handler.read()
            return
        elif isinstance(stmt, Accept):
            self.execute_accept(self.prog[self.labels[self.current_subroutine][stmt.formatLabel]], stmt.args)
            return
        elif isinstance(stmt, Dimension):
            for arg in stmt.args:
                # hack
                if arg[0] == 'RTEXT' or arg[0] == 'LLINE':
                    continue

                # If Implicit statement exists
                # specifing first letter of variable use Implicit to type variable
                defaultVal = 0.0
                if self.varstack[-1].get('IMPLICIT-INT',None) != None:
                    if arg[0][0] in self.varstack[-1]['IMPLICIT-INT'].vars:
                        defaultVal = 0
                elif self.varstack[-1].get('IMPLICIT-REAL',None) != None:
                    if arg[0][0] in self.varstack[-1]['IMPLICIT-REAL'].vars:
                        defaultVal = 0.0

                self.varstack[-1][arg[0]] = make_dimension(arg[1],defaultVal)
            return
        elif isinstance(stmt, Type):
            format = self.prog[self.labels[self.current_subroutine][stmt.formatLabel]]
            self.execute_type(format, stmt.args)
            return
        elif isinstance(stmt, Implicit):
            # Only works for one IMPLICIT statement of each type per module
            # could be improved to additivly build list of starting variable letters
            if stmt.kind == 'INTEGER':
                self.varstack[-1]['IMPLICIT-INT'] = stmt
            elif stmt.kind == 'REAL':
                self.varstack[-1]['IMPLICIT-REAL'] = stmt
            return
        elif isinstance(stmt, ExplDeclare):
            # Create instance of variable and sets the type
            # Decleration must be prior to variable use in module
            # Could be improved to support default value within // and array dimensions
            # as well as support for more than just INTEGER and REAL types
            for varname in stmt.vars:
                self.varstack[-1][varname] = (0 if stmt.type == 'INTEGER' else 0.0)
            return

        print('halt on', stmt)
        exit()


def make_dimension(size,defaultVal):
    if isinstance(size, tuple):
        a, b = size
        mat = []
        for i in xrange(a):
            mat.append([defaultVal] * b)
        return ["MREAL", (a, b), mat]
    return ["AREAL", size, [defaultVal] * size]

def equals(a, b):
    result = equals_inner(a, b)
    return result

def equals_inner(a, b):
    if a == '' or a == ' ':
        a = 0
    if b == '' or b == ' ':
        b = 0
    if isinstance(a, long):
        a = int(a)
    if isinstance(b, long):
        b = int(b)
    if (isinstance(a, int) or isinstance(a, float)) and \
        (isinstance(b, int) or isinstance(b, float)):
        return a == b
    if isinstance(a, str) and isinstance(b, str):
        x = a.strip() == b.strip()
        return x
    if isinstance(a, str) and isinstance(b, int):
        return string_to_dec_num(a) & 0x7ffffffff == b & 0x7ffffffff # hack
    if isinstance(a, int) and isinstance(b, str):
        return a & 0x7ffffffff == string_to_dec_num(b) & 0x7ffffffff # hack
    if isinstance(a, str) and not isinstance(b, str):
        return False
    if isinstance(b, str) and a == 0:
        return False
    if isinstance(a, str) and b == 0:
        return False
    print('failed to compare "%s" %s "%s" %s' % (a, type(a), b, type(b)))
    exit()

def string_to_dec_num(text):
    base = 128
    maxl = 5
    offset = 0
    bitshift = 1
    while len(text) < maxl:
        text += ' '
    x = [ord(t) - offset for t in text]

    s = [n * (base**(maxl - i - 1)) for i, n in enumerate(x)]
    return sum(s) * (1 << bitshift)

def to_string(num):
    if isinstance(num, str):
        return num
    # this probably is not a word
    if num < 536870912:
        return str(num)

    base = 128
    maxl = 5
    offset = 0
    bitshift = 1
    num = int(num / 2)
    # sometimes this is removed because of a hack
    num |= 0x400000000
    return ''.join(chr((num >> (i * 7)) & 0x7f) for i in xrange(4, -1, -1))

def wrap(x):
    if isinstance(x, float):
        return x
    if x > 0:
        return x & 0x7ffffffff
    if x < 0:
        return -((2**35) - (x % (2**35)))
    return x

class StdHandler(object):
    def __init__(self):
        pass
    def write(self, data):
        sys.stdout.write(data)
        sys.stdout.flush()
    def read(self):
        return raw_input()

if __name__ == '__main__':
    handler = StdHandler()
    game = Game(handler)
    game.go()
