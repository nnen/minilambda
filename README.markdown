MiniLambda Interactive Interpreter
==================================

MiniLambda is a simple lambda calculus-like language.

Usage
-----

Entire interpreter is a single python script, `minilambda.py`. It has been
tested with Python 2.6 and 2.7, but may run with other versions. Most unix-like
operating systems come with Python installed, so all you need to do is to
execute it from terminal:

    $ ./minilambda.py

Syntax
------

MiniLambda allows multiple character identifiers, so `ab` is not the
application of `a` on `b`, it is a single variable `ab`. Whitespace needs to be
put between adjancent identifiers, for example `a b`, or `\x y . x`.

Instead of lambda symbol, MiniLambda uses backslash, `\`. So a lambda function which
takes an argument `x` and returns `x` to `x` is: `\x . x x`.

Subexpressions can be parenthised: `a (b c)` - apply `a` to the result of
application of `b` to `c`.

Application is left-associative. This means applications are evaluated from
left to right, for example `a b c d` is equvialent to `(((a b) c) d)`.

An extension over pure lambda calculus, let expressions allow you to define
value of a variable. For instance: `let T = \t f . t`. In each subsequent
expression, all free occurences of variable T will be substituted by `\t f .
t`. Interpreter command `dump` will print all of your defined variables.

