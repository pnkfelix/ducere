use crate::expr;
use crate::grammar::{Binding, NonTerm, Term};
use crate::blackbox::BlackboxName;

mod tree;

pub use tree::Tree;

// notation from paper: `x:A(v)XXX`,
// e.g. `x:A(v)< T' >`
// e.g. `x:A(v) := w`

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Parsed<X> {
    var: Option<expr::Var>,
    nonterm: NonTerm,
    input: expr::Val,
    payload: X,
}

#[derive(Clone, PartialEq, Eq)]
pub enum AbstractNode<X> {
    Term(Term),
    Binding(Binding),
    BlackBox(BlackboxName),
    Parse(Parsed<X>),
}

// const NONTERM_BRACKETS: (char, char) = ('⟨', '⟩');
const NONTERM_BRACKETS: (char, char) = ('(', ')');

impl<X> AbstractNode<X> {
    fn fmt_map(
        &self,
        w: &mut std::fmt::Formatter,
        f: impl Fn(&X, &mut std::fmt::Formatter) -> std::fmt::Result) -> std::fmt::Result
    {
        match self {
            AbstractNode::Term(t) => { write!(w, "\"{}\"", t.string() ) }
            AbstractNode::Binding(b) => { write!(w, "{{{}:={}}}", (b.0).0, b.1) }
            AbstractNode::BlackBox(bb) => { write!(w, "<{}>", bb.0) }
            AbstractNode::Parse(p) => {
                let bd = NONTERM_BRACKETS;
                match (&p.var, &p.input) {
                    (None, expr::Val::Unit) => {
                        write!(w, "{NT}{b}", NT=p.nonterm.0, b=bd.0)?;
                        f(&p.payload, w)?;
                        write!(w, "{d}", d=bd.1)
                    }
                    (None, input) => {
                        write!(w, "{NT}({IN}){b}", b=bd.0, NT=p.nonterm.0, IN=input)?;
                        f(&p.payload, w)?;
                        write!(w, "{d}", d=bd.1)
                    }
                    (Some(var), expr::Val::Unit) => {
                        write!(w, "{VAR}:{NT}{b}", b=bd.0, VAR=var.0, NT=p.nonterm.0)?;
                        f(&p.payload, w)?;
                        write!(w, "{d}", d=bd.1)
                    }
                    (Some(var), input) => {
                        write!(w, "{VAR}:{NT}({IN}){b}",
                               b=bd.0, VAR=var.0, NT=p.nonterm.0, IN=input)?;
                        f(&p.payload, w)?;
                        write!(w, "{d}", d=bd.1)
                    }
                }
            }
        }
    }
}

impl<X: std::fmt::Debug> std::fmt::Debug for AbstractNode<X> {
    fn fmt(&self, w: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.fmt_map(w, |load, w| write!(w, "{:?}", load))
    }
}

impl<X: std::fmt::Display> std::fmt::Display for AbstractNode<X> {
    fn fmt(&self, w: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.fmt_map(w, |load, w| write!(w, "{}", load))
    }
}

pub(crate) fn nonterminal_free<T>(v: &[AbstractNode<T>]) -> bool {
    for n in v {
        if let AbstractNode::Parse(..) = n {
            return false;
        }
    }
    return true;
}

pub struct Sentential(pub Vec<AbstractNode<()>>);
