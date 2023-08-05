use crate::expr;
use crate::grammar::{Binding, NonTerm, Term};
use crate::node::{AbstractNode, Parsed};

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
        crate::node::nonterminal_free(&self.0)
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
