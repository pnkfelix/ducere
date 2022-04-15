use std::collections::{HashMap};
use crate::{Blackbox, NonTerm, Term};
use crate::expr::{Expr, Var}/*}*/;  // check out the (bad) error you get with that uncommented.

pub enum Action {
    Term(Term),
    Constraint(Expr),
    Binding(Var, Expr),
    Blackbox(Blackbox, Expr),
    NonTerm(Var, NonTerm, Expr),
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

pub struct StateData {
    label: String,
    transitions: Vec<(Action, State)>,
    calls: Vec<(Expr, State)>,
    output_if_final: Option<Vec<NonTerm>>,
}

impl StateData {
    pub fn label(&self) -> &str { &self.label }
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

struct StateBuilder(StateData);

impl StateBuilder {
    fn final_state(nt: String) -> Self {
        Self(StateData {
            label: nt.clone(),
            transitions: vec![],
            calls: vec![],
            output_if_final: Some(vec![NonTerm(nt)]),
        })
    }

    fn labelled(l: String) -> Self {
        Self(StateData {
            label: l,
            transitions: vec![],
            calls: vec![],
            output_if_final: None,
        })
    }

    fn action(mut self, action: Action, next: State) -> Self {
        self.0.transitions.push((action, next));
        self
    }

    fn term(self, term: Term, next: State) -> Self {
        self.action(Action::Term(term), next)
    }

    fn constraint(self, expr: Expr, next: State) -> Self {
        self.action(Action::Constraint(expr), next)
    }

    fn binding(self, var: Var, expr: Expr, next: State) -> Self {
        self.action(Action::Binding(var, expr), next)
    }

    fn non_term(self, var: Var, nt: NonTerm, expr: Expr, next: State) -> Self {
        self.action(Action::NonTerm(var, nt, expr), next)
    }

    fn build(self) -> StateData {
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
