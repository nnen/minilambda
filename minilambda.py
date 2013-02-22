#!/Library/Frameworks/Python.framework/Versions/2.7/bin/python
# -*- coding: utf-8 -*-

import cmd

import ply.lex as lex
import ply.yacc as yacc


class Lexer(object):
    tokens = (
        "KW_LET",

        "EQUALS",
        
        "LAMBDA",
        "IDENT",
        "DOT",
        "LEFT_PAR",
        "RIGHT_PAR",
    )
    
    keywords = {
        "let": "KW_LET",
    }

    t_EQUALS = r'\='
    
    t_LAMBDA = r'\^'
    #t_IDENT = r'[a-zA-Z_+*/-]+'
    t_DOT = r'\.'
    t_LEFT_PAR = r'\('
    t_RIGHT_PAR = r'\)'
    
    t_ignore = " \t"
    
    def t_IDENT(self, t):
        r'[a-zA-Z_+*/-]+'
        t.type = self.keywords.get(t.value, "IDENT")
        return t
    
    def t_newline(self, t):
        r'\n+'
        t.lexer.lineno += t.value.count("\n")
    
    def t_error(self, t):
        print "Illegal character '%s'" % t.value[0]
        t.lexer.skip(1)
    
    def __init__(self):
        self.lexer = lex.lex(module = self)
    
    def scan(self, source):
        self.lexer.input(source)
        t = self.lexer.token()
        while t:
            yield t
            t = self.lexer.token()


class Parser(object):
    tokens = Lexer.tokens
    
    def p_expression(self, t):
        r"""expression : atom"""
        t[0] = t[1]
    
    def p_expression_app(self, t):
        r"""expression : expression atom"""
        t[0] = Application(t[1], (t[2], ))
    
    def p_atom_var_ref(self, t):
        r"""atom : IDENT"""
        t[0] = VarRef(t[1])
    
    def p_atom_lambda(self, t):
        r"""atom : LAMBDA var-list DOT expression"""
        t[0] = Lambda(t[2], t[4])
    
    def p_atom_par(self, t):
        r"""atom : LEFT_PAR expression RIGHT_PAR"""
        t[0] = t[2]
    
    def p_atom_par_let(self, t):
        r"""atom : let"""
        t[0] = t[1]
    
    def p_var_list(self, t):
        r"""var-list : IDENT""" 
        t[0] = [t[1], ]
    
    def p_var_list_rest(self, t):
        r"""var-list : var-list IDENT""" 
        t[0] = t[1] + [t[2], ]
    
    def p_let(self, t):
        r"""let : KW_LET IDENT EQUALS expression"""
        t[0] = LetExp(t[2], t[4])
    
    def __init__(self):
        self.yacc = yacc.yacc(module = self)
    
    def parse(self, source):
        return self.yacc.parse(source, lexer = Lexer().lexer)


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
    def __init__(self, var, value):
        self.var = var
        self.value = value
    
    def __repr__(self):
        return "%s(%r, %r)" % (type(self).__name__, self.var, self.value)
    
    def __str__(self):
        return "let %s = %s" % (self.var, self.value, )
    
    def chain(self, other):
        return Application(Lambda([self.var, ], other), [self.value, ])


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


class Lambda(Expression):
    def __init__(self, variables, expr):
        self.variables = variables
        self.expr = expr
    
    def __repr__(self):
        return "%s(%r, %r)" % (type(self).__name__, self.variables, self.expr, )
    
    def __str__(self):
        return "(^%s . %s)" % (" ".join(self.variables), self.expr, )
    
    def apply(self, args):
        subs = {k: v for (k, v) in zip(self.variables, args)}
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
        expr = self.expr.normalize()
        if expr is None:
            return None
        return Lambda(list(self.variables), expr)


class InteractiveInterpreter(cmd.Cmd):
    def __init__(self):
        cmd.Cmd.__init__(self)
        self.prompt = ">>> "
        self.parser = Parser()
        #self.last_expr = Expression()
        self.lets = []
    
    def do_exit(self, args):
        return True
    
    def do_quit(self, args):
        return True

    def do_dump(self, args):
        print "LETS:"
        for l in self.lets:
            print "\t%s" % (l, )
    
    def default(self, line):
        e = self.parser.parse(line)
        if e:
            if isinstance(e, LetExp):
                self.lets.append(e)
            else:
                for l in reversed(self.lets):
                    e = l.chain(e)
                try:
                    print normalize(e)
                except Exception:
                    print "Normalization failed. Possible loop."
        return False


def main():
    console = InteractiveInterpreter()
    console.cmdloop()
    

if __name__ == '__main__':
    main()

