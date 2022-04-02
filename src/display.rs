use crate::{expr, NonTerm, RegularRightSide, Term};

#[derive(Copy, Clone)]
enum RrsContext { Concat, Either, Kleene, }

impl RegularRightSide {
    fn needs_parens(&self, context: RrsContext) -> bool {
        match (self, context) {
            (RegularRightSide::EmptyString |
             RegularRightSide::EmptyLanguage |
             RegularRightSide::Term(_) |
             RegularRightSide::NonTerm { .. } |
             RegularRightSide::Binding { .. }, _) => false,

            (RegularRightSide::Concat(..), RrsContext::Concat) => false,
            (RegularRightSide::Concat(..), RrsContext::Either | RrsContext::Kleene) => true,

            (RegularRightSide::Either(..), RrsContext::Either) => false,
            (RegularRightSide::Either(..), RrsContext::Concat | RrsContext::Kleene) => true,

            (RegularRightSide::Kleene(_), _) => false,

            (RegularRightSide::Constraint(_), _) => false,
            (RegularRightSide::Blackbox(..), _) => unimplemented!(),
        }
    }
}

impl std::fmt::Display for RegularRightSide {
    fn fmt(&self, w: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            RegularRightSide::EmptyString => write!(w, "''"),
            RegularRightSide::EmptyLanguage => write!(w, "empty"),
            RegularRightSide::Term(Term::C(c)) => write!(w, "'{}'", c),
            RegularRightSide::Term(Term::S(s)) => write!(w, "'{}'", s),
            RegularRightSide::NonTerm { x, A: NonTerm(name), e } => {
                match (x, e) {
                    (None, None) => write!(w, "{}", name),
                    (None, Some(e)) => write!(w, "<{}({})>", name, e),
                    (Some(expr::Var(x)), None) => write!(w, "<{}:={}>", x, name),
                    (Some(expr::Var(x)), Some(e)) => write!(w, "<{}:={}({})>", x, name, e),
                }
            }
            RegularRightSide::Binding { x: expr::Var(x), e } => write!(w, "{{ {} := {} }}", x, e),
            RegularRightSide::Concat(lhs, rhs) => {
                let ctxt = RrsContext::Concat;
                if !lhs.needs_parens(ctxt) && !rhs.needs_parens(ctxt) {
                    write!(w, "{} {}", lhs, rhs)
                } else {
                    write!(w, "({}) ({})", lhs, rhs)
                }
            }
            RegularRightSide::Either(lhs, rhs) => {
                let ctxt = RrsContext::Either;
                if !lhs.needs_parens(ctxt) && !rhs.needs_parens(ctxt) {
                    write!(w, "{} | {}", lhs, rhs)
                } else {
                    write!(w, "({}) | ({})", lhs, rhs)
                }
            }
            RegularRightSide::Kleene(r) => {
                let ctxt = RrsContext::Kleene;
                if r.needs_parens(ctxt) {
                    write!(w, "{}*", r)
                } else {
                    write!(w, "({})*", r)
                }
            }
            RegularRightSide::Constraint(e) => {
                write!(w, "[{}]", e)
            }
            RegularRightSide::Blackbox(_bb, _e) => {
                unimplemented!()
            }
        }
    }
}

mod expr_display {
    use crate::expr::{BinOp, Env, Expr, Val, Var};
    impl std::fmt::Display for BinOp {
        fn fmt(&self, w: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(w, "{}", match *self {
                BinOp::Add => "+",
                BinOp::Sub => "-",
                BinOp::Mul => "*",
                BinOp::Div => "/",
                BinOp::Gt => ">",
                BinOp::Ge => ">=",
                BinOp::Lt => "<",
                BinOp::Le => "<=",
                BinOp::Eql => "==",
                BinOp::Neq => "!=",
            })
        }
    }
    
    impl std::fmt::Display for Val {
        fn fmt(&self, w: &mut std::fmt::Formatter) -> std::fmt::Result {
            match self {
                Val::Bool(b) => write!(w, "{:?}", b),
                Val::Unit => write!(w, "()"),
                Val::String(s) => write!(w, "\"{}\"", s),
                Val::Int(i) => write!(w, "{:?}", i),
            }
        }
    }

    impl Expr {
        fn needs_parens(&self, ctxt: BinOp) -> bool {
            match (self, ctxt) {
                (Expr::Var(_), _) |
                (Expr::Lit(_), _) => false,

                (Expr::BinOp(inner_op, _lhs, _rhs), outer_op) => inner_op != &outer_op,
            }
        }
    }

    impl std::fmt::Display for Expr {
        fn fmt(&self, w: &mut std::fmt::Formatter) -> std::fmt::Result {
           match self {
                Expr::Var(Var(v)) => write!(w, "{}", v),
                Expr::Lit(val) => write!(w, "{}", val),

                Expr::BinOp(op, lhs, rhs) => {
                    match (lhs.needs_parens(*op), rhs.needs_parens(*op)) {
                        (true, true) => 
                            write!(w, "({}) {} ({})", lhs, op, rhs),
                        (true, false) => 
                            write!(w, "({}) {} {}", lhs, op, rhs),
                        (false, true) => 
                            write!(w, "{} {} ({})", lhs, op, rhs),
                        (false, false) => 
                            write!(w, "{} {} {}", lhs, op, rhs),
                    }
                }
            }
        }
    }

    impl std::fmt::Display for Env {
        fn fmt(&self, w: &mut std::fmt::Formatter) -> std::fmt::Result {
            let mut content: String = self.bindings()
                .map(|(Var(x), val)| format!("{}={},", x, val))
                .collect();
            // drop last comma, if any content was added at all.
            content.pop();
            write!(w, "[{}]", content)
        }
    }
}
