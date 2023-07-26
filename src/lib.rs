 #[macro_use] extern crate lalrpop_util;

pub type Spanned<Tok, Loc, Error> = Result<(Loc, Tok, Loc), Error>;

mod luthor;
pub mod transducer;
pub mod earley;

pub use toyman::{Lexer, Tok};

// identity macro that is one character away from dbg.
macro_rules! nbg {
    ($x:expr) => { $x }
}

mod toyman;

mod grammar;

use grammar::{Grammar, Rule, RegularRightSide, Term, NonTerm, Binding};

mod blackbox;

use blackbox::{BlackboxName};

mod display;
mod rendering;
mod util;

trait ParseMatches {
    fn has_parse(&mut self) -> bool;
    fn no_parse(&mut self) -> bool { ! self.has_parse() }
}

impl<'s> ParseMatches for Box<dyn Iterator<Item=expr::Env> + 's> {
    fn has_parse(&mut self) -> bool { self.next().is_some() }
}

mod specification_rules;

#[cfg(test)]
mod tests;
pub mod expr;

lalrpop_mod!(pub yakker); // synthesized by LALRPOP

#[derive(PartialEq, Debug)]
pub enum YakkerError {
    NoCharAfterBackslash,
    UnrecognizedChar(char),
    Lex(luthor::LexicalError),
}

fn normalize_escapes(input: &str) -> Result<String, YakkerError> {
    let mut s = String::with_capacity(input.len());
    let mut cs = input.chars();
    while let Some(c) = cs.next() {
        if c == '\\' {
            match cs.next() {
                None => return Err(YakkerError::NoCharAfterBackslash),
                Some(c @ '\\') | Some(c @ '"') => { s.push(c); continue }
                Some('n') => { s.push('\n'); continue }
                Some('t') => { s.push('\t'); continue }
                Some('r') => { s.push('\r'); continue }
                Some(c) => return Err(YakkerError::UnrecognizedChar(c)),
           }
        } else {
            s.push(c);
        }
    }
    return Ok(s);
}

// #[derive(PartialEq, Eq, Clone, Debug)]
// pub struct Var(String);
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct Val(String);

impl From<crate::expr::Val> for Val {
    #[track_caller]
    fn from(v: crate::expr::Val) -> Val {
        if let crate::expr::Val::String(s) = v {
            Val(s)
        } else {
            panic!("tried to convert non-string expressed value {:?} into a string-val", v);
        }
    }
}

impl From<char> for Term { fn from(a: char) -> Self { Self::C(a.into()) } }
impl From<&str> for Term { fn from(a: &str) -> Self { Self::S(normalize_escapes(a.into()).unwrap()) } }
impl From<&str> for NonTerm { fn from(a: &str) -> Self { Self(a.into()) } }
impl From<&str> for Val { fn from(v: &str) -> Self { Self(v.into()) } }

// notation from paper: `x:A(v)XXX`,
// e.g. `x:A(v)< T' >`
// e.g. `x:A(v) := w`

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Parsed<X> { var: Option<expr::Var>, nonterm: NonTerm, input: expr::Val, payload: X }

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

fn nonterminal_free<T>(v: &[AbstractNode<T>]) -> bool {
    for n in v {
        if let AbstractNode::Parse(..) = n {
            return false;
        }
    }
    return true;
}

pub struct Sentential(pub Vec<AbstractNode<()>>);

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Tree(pub Vec<AbstractNode<Tree>>);

#[cfg(not_now)]
impl std::fmt::Debug for Tree {
    fn fmt(&self, w: &mut std::fmt::Formatter) -> std::fmt::Result {
        WITHIN_TREE.with(|within_tree| {
            {
                let mut within_tree = within_tree.borrow_mut();
                if !*within_tree {
                    write!(w, "Tree")?;
                }
                *within_tree = true;
            }
            write!(w, "(")?;
            for node in &self.0 {
                write!(w, "{:?}", node)?;
            }
            {
                let mut within_tree = within_tree.borrow_mut();
                *within_tree = false;
            }
            write!(w, ")")
        })
    }
}

thread_local! {
    pub static WITHIN_TREE: std::cell::RefCell<bool> = std::cell::RefCell::new(false);
}

impl Tree {
    pub fn extend_term(&mut self, term: Term) {
       self.0.push(AbstractNode::Term(term));
    }

    pub fn extend_bind(&mut self, x: expr::Var, v: expr::Val) {
        self.0.push(AbstractNode::Binding(Binding(x, v)));
    }

    pub fn extend_parsed(&mut self, x: Option<expr::Var>, nt: NonTerm, v: expr::Val, t: Tree) {
        self.0.push(AbstractNode::Parse(Parsed {
            var: x, nonterm: nt, input: v.into(), payload: t,
        }));
    }
}

impl From<&str> for Tree {
    fn from(s: &str) -> Tree {
        Tree(s.chars().map(|c|AbstractNode::Term(Term::C(c))).collect())
    }
}

// W : AbstractString
// m : AbstractString that is nonterminal-free
pub struct AbstractString(Vec<AbstractNode<String>>);

impl AbstractString {
    // detetermines if W can be treated as an m.
    pub fn nonterminal_free(&self) -> bool {
        nonterminal_free(&self.0)
    }

    // ||W|| from paper
    pub fn erased(&self) -> String {
        let mut accum = String::new();
        for n in &self.0 {
            let backing: String;
            let s: &str = match n {
                AbstractNode::Term(Term::S(s)) => s,
                AbstractNode::Term(Term::C(c)) => { backing = [c].into_iter().collect(); &backing }
                AbstractNode::Binding(_) => continue,
                AbstractNode::BlackBox(bb) => &bb.0,
                AbstractNode::Parse(p) => &p.payload,
            };
            accum.push_str(s);
        }
        accum
    }

    // Strings(W, A, v) from paper
    pub fn strings(&self, a: NonTerm, v: expr::Val) -> Vec<&str> {
        let mut accum: Vec<&str> = Vec::new();
        for n in &self.0 {
            if let AbstractNode::Parse(p) = n {
                if p.nonterm == a && p.input == v {
                    accum.push(&p.payload);
                }
            }
        }
        accum
    }
}

impl Tree {
    pub fn leaves<'a>(&'a self) -> Vec<std::borrow::Cow<'a, str>> {
        use std::borrow::Cow;
        let mut accum: Vec<Cow<str>> = Vec::new();
        for n in &self.0  {
            match n {
                AbstractNode::Term(Term::S(s)) => accum.push(s.into()),
                AbstractNode::Term(Term::C(c)) => {
                    let mut s = String::new();
                    s.push(*c);
                    accum.push(s.into());
                }
                AbstractNode::Binding(_) => continue,
                AbstractNode::BlackBox(bb) => accum.push((&bb.0).into()),
                AbstractNode::Parse(p) => {
                    accum.extend(p.payload.leaves().into_iter())
                }
            }
        }
        accum
    }
    pub fn roots(&self) -> AbstractString {
        let mut accum: Vec<AbstractNode<String>>= Vec::new();
        for n in &self.0  {
            match n {
                AbstractNode::Term(t) => accum.push(AbstractNode::Term(t.clone())),
                AbstractNode::Binding(b) => accum.push(AbstractNode::Binding(b.clone())),
                AbstractNode::BlackBox(bb) => accum.push(AbstractNode::BlackBox(bb.clone())),
                AbstractNode::Parse(Parsed{ var, nonterm, input, payload }) => {
                    let var = var.clone();
                    let nonterm = nonterm.clone();
                    let input = input.clone();
                    let mut leaves = String::new();
                    for leaf in payload.leaves() {
                        leaves.push_str(&leaf);
                    }
                    let payload = leaves;
                    accum.push(AbstractNode::Parse(Parsed {
                        var, nonterm, input, payload }));
                }
            }
        }
        AbstractString(accum)
    }
}
    
