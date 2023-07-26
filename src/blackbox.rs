use crate::expr;
use crate::Term;

pub trait Recognizer {
    type Term;
    type String;
    fn accept(&self, iter: &mut dyn Iterator<Item=&Self::Term>) -> Option<Self::String>;
}

#[derive(Clone, PartialEq, Eq)]
pub struct Blackbox {
    name: String,
    pub(crate) from_val: fn(expr::Val) -> Box<dyn Recognizer<Term=Term, String=String>>,
}

impl std::fmt::Debug for Blackbox {
    fn fmt(&self, w: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(w, "blackbox[{}]", self.name)
    }
}

// notation from paper: `< w >`
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct BlackboxName(pub(crate) String);

