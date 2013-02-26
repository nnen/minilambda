#!/usr/bin/python
#!/Library/Frameworks/Python.framework/Versions/2.7/bin/python
# -*- coding: utf-8 -*-

import cmd
import re
import sys
from collections import namedtuple


__doc__ = """minilambda - MiniLambda Interactive Interpreter

MiniLambda is a simple lambda calculus-like language.

Syntax:

MiniLambda allows multiple character identifiers, so `ab` is not the
application of `a` on `b`, it is a single variable `ab`. Whitespace needs to be
put between adjancent identifiers, for example `a b`, or `\\x y . x`.

Instead of lambda symbol, MiniLambda uses caret, `\\`. So a lambda function which
takes an argument `x` and returns `x` to `x` is: `\\x . x x`.

Subexpressions can be parenthised: `a (b c)` - apply `a` to the result of
application of `b` to `c`.

Application is left-associative. This means applications are evaluated from
left to right, for example `a b c d` is equvialent to `(((a b) c) d)`.
"""


Token = namedtuple("Token", "type value line column pos")


def _tokens(lexer, *tokens):
    result = []
    for name, exp, func in tokens:
        meth_name = "t_%s" % (name, )
        func = getattr(lexer, meth_name, None)
        result.append((name, re.compile(exp), func ))
    return result


class Lexer(object):
    TOKENS = (
        ("KW_LET",    r"let",               None, ),
        #("KW_PRINT",  r"print",             None, ),
        
        ("EQUALS",    r'\=',                None, ), 
        
        ("LAMBDA",    r'\\',                None, ),
        ("IDENT",     r'[0-9a-zA-Z_+*/-]+', None, ),
        ("DOT",       r'\.',                None, ),
        ("LEFT_PAR",  r'\(',                None, ),
        ("RIGHT_PAR", r'\)',                None, ),
        
        ("COMMA",     r'\,',                None, ),
        
        ("newline",   r'\n+',               None, ),
    )
    
    UNKNOWN_TOKEN = "UNKNOWN"
    
    IGNORE = " \t"
    DELIMITERS = " \t\n()\\."
    
    def t_newline(self, t):
        self.line += t.value.count("\n")
        self.column = 1
        return None
    
    def __init__(self, s = None):
        self.tokens = _tokens(self, *self.TOKENS)
        if s:
            self.input(s)
    
    def __iter__(self):
        return self
    
    def input(self, s):
        self.data = s
        self.pos = 0
        self.line = 1
        self.column = 1
    
    def token(self):
        data = self.data
        pos = self.pos
        
        while pos < len(data):
            if data[pos] in self.IGNORE:
                pos += 1
                continue
            
            for name, exp, func in self.tokens:
                m = exp.match(data, pos)
                if not m: continue
                
                tok = Token(name, m.group(), self.line, self.column, pos)
                self.column += len(tok.value)
                
                if func:
                    tok = func(tok)
                
                pos = m.end()
                self.pos = pos
                
                if not tok:
                    break
                return tok
            else:
                end_pos = pos
                while (end_pos < len(data)) and (data[end_pos] not in self.DELIMITERS):
                    end_pos += 1
                
                if end_pos >= len(data):
                    return None
                
                err_tok = Token(self.UNKNOWN_TOKEN,
                                data[pos:end_pos],
                                self.line,
                                self.column,
                                pos)
                
                pos = end_pos
                self.pos = pos
                self.column += end_pos - pos
                return err_tok
        
        return None
    
    def next(self):
        t = self.token()
        if not t:
            raise StopIteration()
        return t
    
    @classmethod
    def runmain(cls):
        lexer = cls(sys.stdin.read())
        for t in lexer:
            print t


class Result(namedtuple("Result", "value rest")):
    def __nonzero__(self):
        return True
    
    def bind(self, parser):
        return parser(self.rest)
    
    def map(self, fn):
        if isinstance(self.value, tuple) and not isinstance(self.value, Token):
            return Result(fn(*self.value), self.rest)
        return Result(fn(self.value), self.rest)


class NoResult(object):
    value = None
    
    def __nonzero__(self):
        return False
    
    def bind(self, parser):
        return self
    
    def map(self, fn):
        return self

NO_RESULT = NoResult()


class MonadicParser(object):
    def prepend(self, other):
        other = MonadicParser.make(other)
        return self.and_(other).map(lambda a, b: [a, ] + b)
    
    def loop(self):
        p = FutureParser()
        p += self.prepend(p).or_([])
        return p
    
    def map(self, fn):
        return FnParser(lambda t: self(t).map(fn))
    
    def or_(self, other):
        other = MonadicParser.make(other)
        return FnParser(lambda t: self(t) or other(t))
    
    def and_(self, other):
        return AndParser(self, MonadicParser.make(other))
    
    def then_(self, other):
        other = MonadicParser.make(other)
        return FnParser(lambda t: self(t).bind(other))
    
    def end(self):
        def fn(tokens):
            result = self(tokens)
            if len(result.rest) > 0:
                return NO_RESULT
            return result
        return FnParser(fn)
    
    def parse(self, tokens):
        return (None, tokens)

    def __call__(self, tokens):
        return self.parse(tokens)

    @classmethod
    def identity(self, value):
        return FnParser(lambda t: Result(value, t))
    
    @classmethod
    def make(self, v):
        if isinstance(v, MonadicParser):
            return v
        if isinstance(v, basestring):
            return TokenParser(v)
        return self.identity(v)


class FnParser(MonadicParser):
    def __init__(self, fn):
        self.fn = fn
    
    def parse(self, tokens):
        return self.fn(tokens)


class TokenParser(MonadicParser):
    def __init__(self, token_type):
        self.token_type = token_type
    
    def parse(self, tokens):
        if (len(tokens) > 0) and (tokens[0].type == self.token_type):
            return Result(tokens[0], tokens[1:])
        return NO_RESULT


class UnaryParser(MonadicParser):
    def __init__(self, inner_parser):
        self.inner_parser = inner_parser


class BinaryParser(MonadicParser):
    def __init__(self, a, b):
        self.a = a
        self.b = b


class AndParser(BinaryParser):
    def parse(self, tokens):
        result = self.a.parse(tokens)
        result2 = result.bind(self.b)
        return result2.bind(lambda t: Result((result.value, result2.value), result2.rest))


class FutureParser(UnaryParser):
    def __init__(self, inner_parser = None):
        UnaryParser.__init__(self, inner_parser)
    
    def __iadd__(self, other):
        self.inner_parser = other
        return self
    
    def parse(self, tokens):
        return self.inner_parser.parse(tokens)


p = MonadicParser.make

TERM_PARSER = FutureParser()
SIMPLE_TERM_PARSER = FutureParser()

VAR_PARSER = p("IDENT").map(lambda t: VarRef(t.value))

VAR_LIST_PARSER = p("IDENT").map(lambda t: t.value).prepend(p("IDENT").map(lambda t: t.value).loop())
LAMBDA_PARSER = p("LAMBDA").then_(VAR_LIST_PARSER).and_(p("DOT").then_(TERM_PARSER)).map(lambda a, b: Lambda(a, b))

PAR_PARSER = p("LEFT_PAR").then_(TERM_PARSER).and_("RIGHT_PAR").map(lambda a, b: a)

LET_PARSER = p("KW_LET").then_("IDENT").and_("EQUALS").map(lambda a, b: a).and_(TERM_PARSER).map(lambda a, b: LetExp(a.value, b))

PRINT_PARSER = p("KW_PRINT").map(lambda t: PrintBuiltin())

#SIMPLE_TERM_PARSER += VAR_PARSER.or_(LAMBDA_PARSER).or_(PAR_PARSER).or_(LET_PARSER).or_(PRINT_PARSER)
SIMPLE_TERM_PARSER += VAR_PARSER.or_(LAMBDA_PARSER).or_(PAR_PARSER).or_(LET_PARSER)

TERM_PARSER += SIMPLE_TERM_PARSER.prepend(SIMPLE_TERM_PARSER.loop()).map(lambda t: Application.from_list(t))

PROGRAM_PARSER = FutureParser()
PROGRAM_PARSER += LET_PARSER.and_(p("COMMA").then_(PROGRAM_PARSER)).map(lambda a, b: a.with_expr(b)).or_(
    TERM_PARSER.and_(p("COMMA").then_(PROGRAM_PARSER)).map(lambda a, b: Sequence([a, b]))).or_(TERM_PARSER)


class Parser(object):
    def parse_term(self, s):
        return TERM_PARSER(list(Lexer(s)))

    def parse_program(self, s):
        return PROGRAM_PARSER(list(Lexer(s)))


def normalize(expr):
    for i in range(1000):
        next_expr = expr.normalize()
        if next_expr is None:
            return expr
        expr = next_expr
    raise Exception("Too many normalization iterations. Possible loop.") 


class Expression(object):
    @property
    def can_be_evaled(self):
        return False
    
    def __repr__(self):
        return "%s()" % (type(self).__name__, )
    
    def __str__(self):
        return "???"
    
    def substitute(self, subs):
        return self
    
    def apply(self, args):
        #return Application(self, args)
        return None
    
    def eval(self):
        return self
    
    def normalize(self):
        return None
    
    def chain(self, other):
        return other


class LetExp(Expression):
    def __init__(self, var, value, expr = None):
        self.var = var
        self.value = value
        self.expr = expr
    
    def __repr__(self):
        if self.expr:
            return "%s(%r, %r, %r)" % (type(self).__name__, self.var, self.value, self.expr)
        return "%s(%r, %r)" % (type(self).__name__, self.var, self.value)
    
    def __str__(self):
        if self.expr:
            return "let %s = %s, %s" % (self.var, self.value, self.expr, )
        return "let %s = %s" % (self.var, self.value, )
    
    def with_expr(self, expr):
        return LetExp(self.var, self.value, expr)
    
    def chain(self, other):
        return Application(Lambda([self.var, ], other), [self.value, ])
    
    def normalize(self):
        return Application(Lambda([self.var, ], self.expr), [self.value, ])


class VarRef(Expression):
    def __init__(self, ident):
        self.ident = ident
    
    def __repr__(self):
        return "%s(%r)" % (type(self).__name__, self.ident, )
    
    def __str__(self):
        return self.ident
    
    def substitute(self, subs):
        try:
            return subs[self.ident]
        except KeyError:
            return self


class Application(Expression):
    @property
    def can_be_evaled(self):
        if reduce(lambda a, b: a and b, map(lambda v: not v.can_be_evaled, self.args)):
            return True
        return False
    
    def __init__(self, fn, args):
        self.fn = fn
        self.args = args
    
    def __repr__(self):
        return "%s(%r, %r)" % (type(self).__name__, self.fn, self.args, )
    
    def __str__(self):
        return "(%s %s)" % (self.fn, ", ".join((str(a) for a in self.args)), )
    
    def substitute(self, subs):
        return Application(
            self.fn.substitute(subs),
            map(lambda v: v.substitute(subs), self.args),
        )
    
    def eval(self):
        fn = self.fn.eval()
        args = map(lambda a: a.eval(), self.args)
        return fn.apply(args)
    
    def normalize(self):
        fn = self.fn.normalize()
        args = map(lambda a: a.normalize(), self.args)
        args_normalized = reduce(lambda a, b: (a or b) is not None, args, None)
        
        if (fn is not None) or args_normalized:
            return Application(fn or self.fn, map(lambda a, b: a or b, args, self.args))
        
        return self.fn.apply(self.args)
    
    @classmethod
    def from_list(self, terms):
        if len(terms) == 1:
            return terms[0]
        #return Application(terms[0], terms[1:])
        return Application(Application.from_list(terms[:-1]), terms[-1:])


class Lambda(Expression):
    def __init__(self, variables, expr):
        self.variables = variables
        self.expr = expr
    
    def __repr__(self):
        return "%s(%r, %r)" % (type(self).__name__, self.variables, self.expr, )
    
    def __str__(self):
        return "(\\%s . %s)" % (" ".join(self.variables), self.expr, )
    
    def apply(self, args):
        subs = {}
        for k, v in zip(self.variables, args):
            subs[k] = v
        #subs = {k: v for (k, v) in zip(self.variables, args)}
        if len(args) < len(self.variables):
            return Lambda(self.variables[len(args):], self.expr.substitute(subs))
        else:
            return self.expr.substitute(subs)
    
    def substitute(self, subs):
        new_subs = dict(subs)
        for v in self.variables:
            new_subs.pop(v, None)
        return Lambda(list(self.variables), self.expr.substitute(new_subs))
    
    def normalize(self):
        if len(self.variables) > 1:
            result = self.expr
            for v in reversed(self.variables):
                result = Lambda([v, ], result)
            return result
        
        expr = self.expr.normalize()
        if expr is None:
            return None
        return Lambda(list(self.variables), expr)


class Sequence(Expression):
    def __init__(self, terms):
        self.terms = terms
    
    def __repr__(self):
        return "%s(%r)" % (type(self).__name__, self.terms, )
    
    def __str__(self):
        return ", ".join([str(t) for t in self.terms])
    
    def substitute(self, subs):
        return Sequence([t.substitute(subs) for t in self.terms])
    
    def normalize(self):
        if len(self.terms) == 1:
            return self.terms[0]
        
        t = self.terms[0].normalize()
        if t:
            return Sequence([t, ] + self.terms[1:])
        return Sequence(self.terms[1:])


#class PrintBuiltin(Expression):
#    def __str__(self):
#        return "print"
#    
#    def apply(self, args):
#        t = args[0].normalize()
#        if not t:
#            print args[0]
#        return Application.from_list(list(args))


class InteractiveInterpreter(cmd.Cmd):
    def __init__(self, **kwargs):
        cmd.Cmd.__init__(self,
                         stdin = kwargs.get("stdin", None),
                         stdout = kwargs.get("stdout", None))
        self.prompt = ">>> "
        #self.parser = Parser()
        self.parser = TERM_PARSER
        self.lets = []
    
    def _println(self, s):
        self.stdout.write(str(s))
        self.stdout.write("\n")
    
    def help_syntax(self, *args):
        self._println(__doc__)
    
    def do_exit(self, args):
        return True
    
    def do_quit(self, args):
        return True
    
    def do_dump(self, args):
        self._println("LETS:")
        for l in self.lets:
            self._println("\t%s" % (l, ))
    
    def default(self, line):
        e = self.parser.parse(list(Lexer(line))).value
        if e:
            if isinstance(e, LetExp):
                self.lets.append(e)
            else:
                for l in reversed(self.lets):
                    e = l.chain(e)
                try:
                    self._println(normalize(e))
                except Exception:
                    self._println("Normalization failed. Possible loop.")
        return False


def main():
    console = InteractiveInterpreter()
    console.cmdloop()


if __name__ == '__main__':
    main()

