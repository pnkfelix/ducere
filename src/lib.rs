#[macro_use] extern crate lalrpop_util;

use std::collections::HashSet;

pub type Spanned<Tok, Loc, Error> = Result<(Loc, Tok, Loc), Error>;

mod luthor;
pub mod transducer;

pub use toyman::{Lexer, Tok};

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
                let x = if let Some(x) = dbg!(self.0.next()) { x } else { return None; };
                let (i, x, j) = match x { Ok(x) => dbg!(x), Err(e) => { return Some(Err(YakkerError::Lex(e))); } };
                let c = dbg!(x.data()).chars().next().unwrap();
                let tok = match (*x.data(), x.kind()) {
                    (s, K::Bracket) => match s {
                        "(" => Tok::PAREN_OPEN,
                        ")" => Tok::PAREN_CLOSE,
                        "{" => Tok::CURLY_BRACE_OPEN,
                        "}" => Tok::CURLY_BRACE_CLOSE,
                        "[" => Tok::SQUARE_BRACKET_OPEN,
                        "]" => Tok::SQUARE_BRACKET_CLOSE,
                        s => panic!("unhandled bracket type `{}`", s),
                    },

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
                        dbg!(Tok::QuoteLit(*c1, &s[1..(s.len()-1)], *c2))
                    }

                    (_, K::Space) => {
                        // skip the space and grab next token.
                        continue;
                    }

                    (s, K::Word(Word::Id(_))) if s.chars().next().unwrap().is_uppercase() => {
                        dbg!(Tok::UpperIdent(s))
                    }
                    (s, K::Word(Word::Id(_))) => {
                        assert!(c == '_' || c.is_lowercase());
                        Tok::LowerIdent(s)
                    }
                };
                return Some(Ok(dbg!((i, tok, j))));
            }
        }
    }

    #[allow(non_camel_case_types)]
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub enum Tok<'a> {
        // "("
        PAREN_OPEN,
        // ")"
        PAREN_CLOSE,
        // "{"
        CURLY_BRACE_OPEN,
        // "}"
        CURLY_BRACE_CLOSE,
        // "["
        SQUARE_BRACKET_OPEN,
        // "]"
        SQUARE_BRACKET_CLOSE,

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


/// Tiny DSL I'm using for prototyping Yakker's support for
/// parameteric-nonterminals, inline-bindings, and conditional guards.
///
/// I imagine the full system, that leverages code-geneation, will support Rust
/// (or whatever the target language is) in this context instead.
pub mod expr {
    #[derive(Copy, Clone, PartialEq, Eq, Debug)]
    pub enum BinOp { Add, Sub, Mul, Div, Gt, Ge, Lt, Le, Eql, Neq }

    #[derive(PartialEq, Eq, Clone, Debug)]
    pub struct Var(pub String);

    // pub const Y_0: Var = Var("Y_0".into());
    pub fn y_0() -> Var { Var("Y_0".into()) }

    #[derive(PartialEq, Eq, Clone, Debug)]
    pub enum Expr { Var(Var), Lit(Val), BinOp(BinOp, Box<Expr>, Box<Expr>) }

    #[derive(PartialOrd, Ord, PartialEq, Eq, Clone, Debug)]
    pub enum Val { Bool(bool), Unit, String(String), Int(i64), }

    pub const TRUE: Val = Val::Bool(true);


    impl std::ops::Add<Val> for Val {
        type Output = Val;
        fn add(self, rhs: Val) -> Self {
            match (self, rhs) {
                (Val::Int(lhs), Val::Int(rhs)) => Val::Int(lhs + rhs),
                (Val::String(mut lhs), Val::String(rhs)) => { lhs.push_str(&rhs); Val::String(lhs) }
                (lhs, rhs) => { panic!("invalid inputs for Add: {:?} and {:?}", lhs, rhs); }
            }
        }
    }

    impl std::ops::Sub<Val> for Val {
        type Output = Val;
        fn sub(self, rhs: Val) -> Self {
            match (self, rhs) {
                (Val::Int(lhs), Val::Int(rhs)) => Val::Int(lhs - rhs),
                (lhs, rhs) => { panic!("invalid inputs for Sub: {:?} and {:?}", lhs, rhs); }
            }
        }
    }

    impl std::ops::Mul<Val> for Val {
        type Output = Val;
        fn mul(self, rhs: Val) -> Self {
            match (self, rhs) {
                (Val::Int(lhs), Val::Int(rhs)) => Val::Int(lhs * rhs),
                (lhs, rhs) => { panic!("invalid inputs for Sub: {:?} and {:?}", lhs, rhs); }
            }
        }
    }

    impl std::ops::Div<Val> for Val {
        type Output = Val;
        fn div(self, rhs: Val) -> Self {
            match (self, rhs) {
                (Val::Int(lhs), Val::Int(rhs)) => Val::Int(lhs / rhs),
                (lhs, rhs) => { panic!("invalid inputs for Sub: {:?} and {:?}", lhs, rhs); }
            }
        }
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct Env(Vec<(Var, Val)>);

    impl Env {
        pub fn empty() -> Self { Env(vec![]) }

        pub fn bind(x: Var, v: Val) -> Self { Env(vec![(x, v)]) }

        pub fn extend(&mut self, x: Var, v: Val) -> &Self {
            self.0.retain(|(x_, _v_)| x_ != &x);
            self.0.push((x, v));
            self
        }

        pub fn lookup(&self, x: &Var) -> Option<&Val> {
            for (y, w) in self.0.iter().rev() {
                if x == y {
                    return Some(w);
                }
            }
            None
        }

        pub fn concat(self, e2: Env) -> Self {
            let mut s = self;
            for (x, v) in e2.0.into_iter() {
                s.extend(x, v);
            }
            s
        }

        pub(crate) fn bindings(&self) -> impl Iterator<Item=&(Var, Val)> {
            self.0.iter()
        }
    }

    impl Expr {
        pub fn eval(&self, env: &Env) -> Val {
            match self {
                Expr::Var(x) => env.lookup(x).unwrap_or_else(|| panic!("failed lookup: {:?}", x)).clone(),
                Expr::Lit(v) => v.clone(),
                Expr::BinOp(op, e1, e2) => {
                    let lhs = e1.eval(env);
                    let rhs = e2.eval(env);
                    match op {
                        BinOp::Add => lhs + rhs,
                        BinOp::Sub => lhs - rhs,
                        BinOp::Mul => lhs * rhs,
                        BinOp::Div => lhs / rhs,
                        BinOp::Gt => Val::Bool(lhs > rhs),
                        BinOp::Ge => Val::Bool(lhs >= rhs),
                        BinOp::Lt => Val::Bool(lhs < rhs),
                        BinOp::Le => Val::Bool(lhs <= rhs),
                        BinOp::Eql => Val::Bool(lhs == rhs), 
                        BinOp::Neq => Val::Bool(lhs != rhs),
                   }
                }
            }
        }
    }

    impl From<char> for Var { fn from(c: char) -> Var { Var(c.to_string()) } }
    impl From<&str> for Var { fn from(c: &str) -> Var { Var(c.to_string()) } }
    impl From<String> for Var { fn from(c: String) -> Var { Var(c) } }
    
    impl From<Var> for Expr { fn from(x: Var) -> Expr { Expr::Var(x) } }
    impl From<Val> for Expr { fn from(v: Val) -> Expr { Expr::Lit(v) } }

    impl From<bool> for Val { fn from(b: bool) -> Val { Val::Bool(b) } }
    impl From<()> for Val { fn from((): ()) -> Val { Val::Unit } }
    impl From<String> for Val { fn from(s: String) -> Val { Val::String(s) } }
    impl From<&str> for Val { fn from(s: &str) -> Val { Val::String(s.to_string()) } }
    impl From<i64> for Val { fn from(n: i64) -> Val { Val::Int(n) } }
    impl<'a> From<&[std::borrow::Cow<'a, str>]> for Val {
        fn from(strs: &[std::borrow::Cow<'a, str>]) -> Val {
            Val::String(strs.iter().map(|cow|&**cow).collect::<String>())
        }
    }
    impl From<&[super::Term]> for Val {
        fn from(terms: &[super::Term]) -> Val {
            use super::Term;
            let mut s = String::new();
            for t in terms {
                match t {
                    Term::C(c) => s.push(*c),
                    Term::S(s2) => s.push_str(&s2),
                }
            }
            Val::String(s)
        }
    }

    impl From<bool> for Expr { fn from(b: bool) -> Expr { let v: Val = b.into(); v.into() } }
    impl From<()> for Expr { fn from((): ()) -> Expr { let v: Val = ().into(); v.into() } }
    impl From<String> for Expr { fn from(s: String) -> Expr { let v: Val = s.into(); v.into() } }
    impl From<&str> for Expr { fn from(s: &str) -> Expr { let v: Val = s.into(); v.into() } }

}

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
impl From<&str> for Term { fn from(a: &str) -> Self { Self::S(a.into()) } }
impl From<&str> for NonTerm { fn from(a: &str) -> Self { Self(a.into()) } }
impl From<&str> for Val { fn from(v: &str) -> Self { Self(v.into()) } }

// notation from paper: `x:A(v)XXX`,
// e.g. `x:A(v)< T' >`
// e.g. `x:A(v) := w`

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Parsed<X> { var: Option<expr::Var>, nonterm: NonTerm, input: expr::Val, payload: X }

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum AbstractNode<X> {
    Term(Term),
    Binding(Binding),
    BlackBox(BlackBox),
    Parse(Parsed<X>),
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
    
