 #[macro_use] extern crate lalrpop_util;

use std::collections::HashSet;

pub type Spanned<Tok, Loc, Error> = Result<(Loc, Tok, Loc), Error>;

mod luthor;
pub mod transducer;
pub mod earley;

pub use toyman::{Lexer, Tok};

// identity macro that is one character away from dbg.
macro_rules! nbg {
    ($x:expr) => { $x }
}

mod toyman {
    use super::{luthor, Spanned, YakkerError};

    pub struct Lexer<'a>(luthor::Lexer<'a>);

    impl<'a> Lexer<'a> {
        pub fn new(input: &'a str) -> Self {
            Lexer(luthor::Lexer::new(input))
        }
    }

    impl<'input> Iterator for Lexer<'input> {
        type Item = Spanned<Tok<'input>, usize, YakkerError>;

        fn next(&mut self) -> Option<Self::Item> {
            use luthor::{Delims, Word, Quoted};
            use luthor::TokKind as K;

            loop {
                let x = if let Some(x) = nbg!(self.0.next()) { x } else { return None; };
                let (i, x, j) = match x { Ok(x) => nbg!(x), Err(e) => { return Some(Err(YakkerError::Lex(e))); } };
                let opt_c = nbg!(x.data()).chars().next();
                let tok = match (*x.data(), x.kind()) {
                    (s, K::Bracket) => {
                        Tok::Bracket(s)
                    }

                    (s, K::Word(Word::Com(_))) => {
                        Tok::Commalike(s)
                    }

                    (s, K::Word(Word::Num(_))) => {
                        Tok::Numeric(s)
                    }

                    (s, K::Word(Word::Op(_))) => {
                        match s {
                            _ => Tok::Operative(s)
                        }
                    }

                    (s, K::Quote(Quoted { sharp_count: _, delim: Delims(c1, c2), content: _ })) => {
                        nbg!(Tok::QuoteLit(*c1, s, *c2))
                    }

                    (_, K::Space) => {
                        // skip the space and grab next token.
                        continue;
                    }

                    (s, K::Word(Word::Id(_))) if s.chars().next().unwrap().is_uppercase() => {
                        nbg!(Tok::UpperIdent(s))
                    }
                    (s, K::Word(Word::Id(_))) => {
			let c = opt_c.unwrap();
                        assert!(c == '_' || c.is_lowercase());
                        Tok::LowerIdent(s)
                    }
                };
                return Some(Ok(nbg!((i, tok, j))));
            }
        }
    }

    #[allow(non_camel_case_types)]
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub enum Tok<'a> {
        // "(", ")", "{", "}", "[", "]"
        Bracket(&'a str),

        // r"[a-z_][a-zA-Z_0-9]*"
        LowerIdent(&'a str),
        // r"[A-Z][a-zA-Z_0-9]*"
        UpperIdent(&'a str),
        // r"[1-9][0-9]*|0"
        Numeric(&'a str),

        // ";" or ","
        Commalike(&'a str),

        // r#""(\\"|[^"])*""# => STRING_LIT
        // r"'[^'\\]'"
        QuoteLit(char, &'a str, char),

        // (other)
        Operative(&'a str),
    }
}


pub trait Recognizer {
    type Term;
    type String;
    fn accept(&self, iter: &mut dyn Iterator<Item=&Self::Term>) -> Option<Self::String>;
}

#[derive(Clone, PartialEq, Eq)]
pub struct Blackbox {
    name: String,
    from_val: fn(expr::Val) -> Box<dyn Recognizer<Term=Term, String=String>>,
}

impl std::fmt::Debug for Blackbox {
    fn fmt(&self, w: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(w, "blackbox[{}]", self.name)
    }
}

#[derive(PartialEq, Eq, Debug)]
pub struct Grammar { pub rules: Vec<Rule> }

impl Grammar {
    pub fn nonterms(&self) -> HashSet<NonTerm> {
        // one might argue that this should also scan the right-hand sides of
        // the rules for non-terminals that are otherwise undefined. But I say
        // that grmmars that do that deserve to be considered ill-formed.
        self.rules.iter().map(|r|r.lhs.clone()).collect()
    }
    pub fn terms(&self) -> HashSet<Term> {
        self.rules.iter().flat_map(|r|r.rhs.terms()).collect()
    }
}

impl Grammar {
    pub fn empty() -> Self { Grammar { rules: vec![] } }

    fn rule(&self, nonterm: &NonTerm) -> Option<&Rule> {
        self.rules.iter().find(|r| &r.lhs == nonterm)
    }
}

#[derive(PartialEq, Eq, Debug)]
pub struct Rule {
    label: String,
    lhs: NonTerm,
    param: Option<expr::Var>,
    rhs: RegularRightSide
}

impl Rule {
    #[cfg(test)]
    pub fn new(lhs: NonTerm, param: Option<expr::Var>, rhs: RegularRightSide) -> Rule {
        Rule { label: String::new(), lhs, param, rhs }
    }

    #[cfg(test)]
    fn labelled_new(label: impl Into<String>, lhs: NonTerm, param: Option<expr::Var>, rhs: RegularRightSide) -> Rule {
        Rule { label: label.into(), lhs, param, rhs }
    }
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub enum RegularRightSide {
    EmptyString,
    EmptyLanguage,
    Term(Term),
    #[allow(non_snake_case)]
    NonTerm { x: Option<expr::Var>, A: NonTerm, e: Option<expr::Expr> },
    Binding { x: expr::Var, e: expr::Expr },
    Concat(Box<Self>, Box<Self>),
    Either(Box<Self>, Box<Self>),
    Kleene(Box<Self>),
    Constraint(expr::Expr),
    Blackbox(Blackbox, expr::Expr),
}

trait Bother<'a, T> { fn b_iter(self) -> Box<dyn Iterator<Item=T> + 'a>; }

impl<'a, T:'a> Bother<'a, T> for Option<T> {
    fn b_iter(self) -> Box<dyn Iterator<Item=T>+'a> {
        Box::new(self.into_iter())
    }
}

impl RegularRightSide {
    fn terms(&self) -> Box<dyn Iterator<Item=Term>> {
        match self {
            RegularRightSide::EmptyString |
            RegularRightSide::EmptyLanguage |
            RegularRightSide::NonTerm { .. } |
            RegularRightSide::Binding { .. }  |
            RegularRightSide::Constraint(_) |
            RegularRightSide::Blackbox(..) => None.b_iter(),

            RegularRightSide::Term(t) => Some(t.clone()).b_iter(),

            RegularRightSide::Concat(lhs, rhs) |
            RegularRightSide::Either(lhs, rhs) => Box::new(lhs.terms().chain(rhs.terms())),

            RegularRightSide::Kleene(inner) => inner.terms(),
        }
    }

    fn nullable(&self, assume_nt: &impl Fn(&NonTerm) -> Nullability) -> Nullability {
        use Nullability::{Nullable, NonNullable, Unknown};
        match self {
            RegularRightSide::EmptyString => Nullable,
            RegularRightSide::EmptyLanguage => NonNullable,
            RegularRightSide::NonTerm { A, .. } => assume_nt(A),
            RegularRightSide::Binding { .. }  => Nullable,
            RegularRightSide::Constraint(_) => Nullable,
            RegularRightSide::Blackbox(..) => Unknown,

            RegularRightSide::Term(t) => t.nullable(),

            RegularRightSide::Concat(lhs, rhs) => lhs.nullable(assume_nt).concat(rhs.nullable(assume_nt)),
            RegularRightSide::Either(lhs, rhs) => lhs.nullable(assume_nt).either(rhs.nullable(assume_nt)),

            RegularRightSide::Kleene(_inner) => Nullable,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum Nullability {
    /// Contains empty string.
    Nullable,
    /// Never contains empty string.
    NonNullable,
    /// We do not know if it contains empty string or not.
    Unknown,
}

impl Nullability {
    fn non_null(&self) -> bool {
        *self == Nullability::NonNullable
    }

    fn concat(&self, other: Self) -> Self {
        use Nullability::{Nullable, NonNullable, Unknown};
        match (self, other) {
            (Nullable, Nullable) => Nullable,
            (NonNullable, _) => NonNullable,
            (_, NonNullable) => NonNullable,
            (Nullable, Unknown) => Unknown,
            (Unknown, Nullable) => Unknown,
            (Unknown, Unknown) => Unknown,
        }
    }

    fn either(&self, other: Self) -> Self {
        use Nullability::{Nullable, NonNullable, Unknown};
        match (self, other) {
            (Nullable, _) => Nullable,
            (_, Nullable) => Nullable,
            (NonNullable, NonNullable) => NonNullable,
            (Unknown, NonNullable) => Unknown,
            (NonNullable, Unknown) => Unknown,
            (Unknown, Unknown) => Unknown,
        }
    }
}

impl Term {
    fn nullable(&self) -> Nullability {
        match self {
            Term::C(_) => Nullability::NonNullable,
            Term::S(s) => if s.len() == 0 {
                Nullability::Nullable } else {
                Nullability::NonNullable
            },
        }
    }
}

mod display;
mod rendering;


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

// A grammar G is a tuple (Sigma, Delta, Phi, A_0, R), where
//   Sigma is a finite set of terminals
//   Delta is a finite set of non-terminals
//   Phi si a finite set of blackboxes
//   A_0 in Delta is the start non-terminal, and
//   R maps non-termainsl to regular right sides

// Regular right sides
//
// r ::= epsilon           <empty string>
//    |  empty             <empty language>
//    |  c                 <terminal>
//    |  x := A(e)         <nonterminal>
//    |  x := e            <binding>
//    |  (r r)             <concatenation>
//    |  (r | r)           <alternation>
//    |  (r*)              <Kleene closure>
//    |  [e]               <constraint>
//    |  phi(e)            <blackbox>
//

#[derive(PartialEq, Eq, Clone, Hash, Debug)]
pub enum Term { C(char), S(String) }

impl Term {
    fn string(&self) -> String {
        let mut s = String::new();
        match self {
            Term::C(c) => { s.push(*c) }
            Term::S(s2) => { s = s2.clone(); }
        }
        s
    }
    fn matches(&self, w: &[Term]) -> bool {
        if let (&Term::C(c1), &[Term::C(c2)]) = (self, w) {
            return c1 == c2;
        }
        let left = self.string();
        let left = left.chars().fuse();
        let right: Vec<String> = w.iter().map(|t|t.string()).collect();
        let right = right.iter().map(|s|s.chars()).flatten();
        left.eq(right)
    }
}

#[derive(PartialEq, Eq, Clone, Hash, Debug)]
pub struct NonTerm(String);
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

// notation from paper: `{x := v }`
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct Binding(expr::Var, expr::Val);

// notation from paper: `< w >`
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct BlackBox(String);

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
    BlackBox(BlackBox),
    Parse(Parsed<X>),
}

// const NONTERM_BRACKETS: (char, char) = ('⟨', '⟩');
const NONTERM_BRACKETS: (char, char) = ('(', ')');

impl<X: std::fmt::Debug> std::fmt::Debug for AbstractNode<X> {
    fn fmt(&self, w: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            AbstractNode::Term(t) => { write!(w, "\"{}\"", t.string() ) }
            AbstractNode::Binding(b) => { write!(w, "{{{}:={}}}", (b.0).0, b.1) }
            AbstractNode::BlackBox(bb) => { write!(w, "<{}>", bb.0) }
            AbstractNode::Parse(p) => {
                let bd = NONTERM_BRACKETS;
                match (&p.var, &p.input) {
                    (None, expr::Val::Unit) => {
                        write!(w, "{NT}{b}{LOAD:?}{d}",
                               b=bd.0, d=bd.1, NT=p.nonterm.0, LOAD=p.payload)
                    }
                    (None, input) => {
                        write!(w, "{NT}({IN}){b}{LOAD:?}{d}",
                               b=bd.0, d=bd.1, NT=p.nonterm.0, IN=input, LOAD=p.payload)
                    }
                    (Some(var), expr::Val::Unit) => {
                        write!(w, "{VAR}:{NT}{b}{LOAD:?}{d}",
                               b=bd.0, d=bd.1, VAR=var.0, NT=p.nonterm.0, LOAD=p.payload)
                    }
                    (Some(var), input) => {
                        write!(w, "{VAR}:{NT}({IN}){b}{LOAD:?}{d}",
                               b=bd.0, d=bd.1, VAR=var.0, NT=p.nonterm.0, IN=input, LOAD=p.payload)
                    }
                }
            }
        }
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
    
