use std::collections::{HashMap};
use crate::{Blackbox, NonTerm, Term};
use crate::expr::{Expr, Var}/*}*/;  // check out the (bad) error you get with that uncommented.

#[derive(Debug)]
pub enum Action {
    Term(Term),
    Constraint(Expr),
    Binding(Var, Expr),
    Blackbox(Blackbox, Expr),

    /// `x := A(e); but the common cases are otherwise unreferenced `x` and unit
    /// expr, so we allow those to be omitted rather than force client to make
    /// up inputs.
    NonTerm(Option<Var>, NonTerm, Option<Expr>),
}

pub trait NonTermSpec {
    fn components(self) -> (Option<Var>, NonTerm, Option<Expr>);
}

impl NonTermSpec for NonTerm {
    fn components(self) -> (Option<Var>, NonTerm, Option<Expr>) { (None, self, None) }
}

impl NonTermSpec for (Var, NonTerm) {
    fn components(self) -> (Option<Var>, NonTerm, Option<Expr>) { (Some(self.0), self.1, None) }
}

impl NonTermSpec for (Var, NonTerm, Expr) {
    fn components(self) -> (Option<Var>, NonTerm, Option<Expr>) { (Some(self.0), self.1, Some(self.2)) }
}

impl NonTermSpec for (NonTerm, Expr) {
    fn components(self) -> (Option<Var>, NonTerm, Option<Expr>) { (None, self.0, Some(self.1)) }
}

pub struct Transducer {
    pub(crate) states: HashMap<State, StateData>,
}

impl Transducer {
    pub fn data(&self, state: State) -> &StateData {
        &self.states[&state]
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct State(usize);

#[derive(Debug)]
pub struct StateData {
    label: String,
    param_name: Option<Var>,
    transitions: Vec<(Action, State)>,
    calls: Vec<(Expr, State)>,
    output_if_final: Option<Vec<NonTerm>>,
}

impl StateData {
    pub fn label(&self) -> &str { &self.label }
    pub fn formal_param(&self) -> Option<&Var> { self.param_name.as_ref() }
    pub fn transitions(&self) -> &[(Action, State)] {
        &self.transitions
    }
    pub fn calls(&self) -> &[(Expr, State)] {
        &self.calls
    }
    pub fn output_if_final(&self) -> Option<&[NonTerm]> {
        self.output_if_final.as_ref().map(|v| &v[..])
    }
}

pub struct StateBuilder(StateData);

impl StateBuilder {
    pub fn final_state(nt: String) -> Self {
        Self(StateData {
            label: nt.clone(),
            param_name: None,
            transitions: vec![],
            calls: vec![],
            output_if_final: Some(vec![NonTerm(nt)]),
        })
    }

    pub fn labelled(l: String) -> Self {
        Self(StateData {
            label: l,
            param_name: None,
            transitions: vec![],
            calls: vec![],
            output_if_final: None,
        })
    }

    pub fn parameterized(l: String, var: Var) -> Self {
        Self(StateData {
            label: l,
            param_name: Some(var),
            transitions: vec![],
            calls: vec![],
            output_if_final: None,
        })
    }

    pub fn action(mut self, action: Action, next: State) -> Self {
        self.0.transitions.push((action, next));
        self
    }

    pub fn term(self, term: impl Into<Term>, next: State) -> Self {
        self.action(Action::Term(term.into()), next)
    }

    pub fn term_range(self, range_incl: (char, char), next: State) -> Self {
        let mut s = self;
        let (start, finis) = range_incl;
        for c in start..=finis {
            s = s.action(Action::Term(Term::C(c)), next);
        }
        s
    }

    pub fn constraint(self, expr: Expr, next: State) -> Self {
        self.action(Action::Constraint(expr), next)
    }

    pub fn binding(self, var: Var, expr: Expr, next: State) -> Self {
        self.action(Action::Binding(var, expr), next)
    }

    pub fn non_term(self, nt: impl NonTermSpec, next: State) -> Self {
        let (var, nt, expr) = nt.components();
        self.action(Action::NonTerm(var, nt, expr), next)
    }

    pub fn call(mut self, expr: Expr, next: State) -> Self {
        self.0.calls.push((expr, next));
        self
    }

    pub fn build(self) -> StateData {
        self.0
    }
}

// This is a trick: we have the source code for our tests under a single
// `src/tests/` subdirectory, but we declare it as a module *here*, under this
// module. That way, it has access to private constructors and state that a
// sibling (or in this case, nibling) module would not have access to.
#[cfg(test)]
#[path = "tests/transducer.rs"]
mod tests_for_transducer;
