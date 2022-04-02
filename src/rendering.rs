use crate::{expr, Term};

pub trait Rendered {
    fn rendered(&self) -> String;
}

impl Rendered for expr::Env {
    fn rendered(&self) -> String {
        format!("{}", self)
    }
}

impl Rendered for Option<expr::Env> {
    fn rendered(&self) -> String {
        match self {
            Some(env) => format!("{}", env),
            None => "nil".to_string(),
        }
    }
}

impl Rendered for [Term] {
    fn rendered(&self) -> String {
        self.iter()
            .map(|t| {
                match t {
                    Term::C(c) => c.to_string(),
                    Term::S(s) => s.to_string(),
                }
            })
            .collect()
    }
}
