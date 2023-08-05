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

use grammar::{Grammar, Rule, RegularRightSide, Term, NonTerm};

mod blackbox;

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
mod expr;
mod node;

use node::{AbstractNode, Tree};

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
