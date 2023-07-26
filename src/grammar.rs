use crate::blackbox;
use crate::expr;
use crate::util::Bother;

use std::collections::HashSet;

// A grammar G is a tuple (Sigma, Delta, Phi, A_0, R), where
//   Sigma is a finite set of terminals
//   Delta is a finite set of non-terminals
//   Phi si a finite set of blackboxes
//   A_0 in Delta is the start non-terminal, and
//   R maps non-termainsl to regular right sides


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

    pub fn rule(&self, nonterm: &NonTerm) -> Option<&Rule> {
        self.rules.iter().find(|r| &r.lhs == nonterm)
    }
}

#[derive(PartialEq, Eq, Debug)]
pub struct Rule {
    pub(crate) label: String,
    pub(crate) lhs: NonTerm,
    pub(crate) param: Option<expr::Var>,
    pub(crate) rhs: RegularRightSide
}

impl Rule {
    #[cfg(test)]
    pub fn new(lhs: NonTerm, param: Option<expr::Var>, rhs: RegularRightSide) -> Rule {
        Rule { label: String::new(), lhs, param, rhs }
    }

    #[cfg(test)]
    pub(crate) fn labelled_new(label: impl Into<String>, lhs: NonTerm, param: Option<expr::Var>, rhs: RegularRightSide) -> Rule {
        Rule { label: label.into(), lhs, param, rhs }
    }
}

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
    Blackbox(blackbox::Blackbox, expr::Expr),
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

    pub(crate) fn nullable(&self, assume_nt: &impl Fn(&NonTerm) -> Nullability) -> Nullability {
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
pub(crate) enum Nullability {
    /// Contains empty string.
    Nullable,
    /// Never contains empty string.
    NonNullable,
    /// We do not know if it contains empty string or not.
    Unknown,
}

impl Nullability {
    pub(crate) fn non_null(&self) -> bool {
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

#[derive(PartialEq, Eq, Clone, Hash, Debug)]
pub enum Term { C(char), S(String) }

impl Term {
    pub(crate) fn string(&self) -> String {
        let mut s = String::new();
        match self {
            Term::C(c) => { s.push(*c) }
            Term::S(s2) => { s = s2.clone(); }
        }
        s
    }
    pub fn matches(&self, w: &[Term]) -> bool {
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

#[derive(PartialEq, Eq, Clone, Hash, Debug)]
pub struct NonTerm(pub(crate) String);

// notation from paper: `{x := v }`
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct Binding(pub(crate) expr::Var, pub(crate) expr::Val);

