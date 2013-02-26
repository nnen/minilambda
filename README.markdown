minilambda - MiniLambda Interactive Interpreter
===============================================

MiniLambda is a simple lambda calculus-like language.

Syntax
------

MiniLambda allows multiple character identifiers, so `ab` is not the
application of `a` on `b`, it is a single variable `ab`. Whitespace needs to be
put between adjancent identifiers, for example `a b`, or `\\x y . x`.

Instead of lambda symbol, MiniLambda uses caret, `\\`. So a lambda function which
takes an argument `x` and returns `x` to `x` is: `\\x . x x`.

Subexpressions can be parenthised: `a (b c)` - apply `a` to the result of
application of `b` to `c`.

Application is left-associative. This means applications are evaluated from
left to right, for example `a b c d` is equvialent to `(((a b) c) d)`.

