/// Tiny DSL I'm using for prototyping Yakker's support for
/// parameteric-nonterminals, inline-bindings, and conditional guards.
///
/// I imagine the full system, that leverages code-geneation, will support Rust
/// (or whatever the target language is) in this context instead.

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum BinOp { Add, Sub, Mul, Div, Gt, Ge, Lt, Le, Eql, Neq }

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum UnOp { String2Int }

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub struct Var(pub String);

// pub const Y_0: Var = Var("Y_0".into());
pub fn y_0() -> Var { Var("Y_0".into()) }

#[derive(PartialEq, Eq, Clone, Debug)]
pub enum Expr { Var(Var), Lit(Val), BinOp(BinOp, Box<Expr>, Box<Expr>), UnOp(UnOp, Box<Expr>) }

#[derive(PartialOrd, Ord, PartialEq, Eq, Hash, Clone, Debug)]
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

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Env(Vec<(Var, Val)>);

impl std::fmt::Debug for Env {
    fn fmt(&self, w: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(w, "[")?;
        for (var, val) in &self.0 {
            write!(w, "{:?}={:?}", var, val)?
        }
        write!(w, "]")
    }
}

impl Env {
    pub fn empty() -> Self { Env(vec![]) }

    pub fn bind(x: Var, v: Val) -> Self { Env(vec![(x, v)]) }

    pub fn extend(&mut self, x: Var, v: Val) -> &mut Self {
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

// FIXME: eval should return a Result<Val, EvalError>, or something. We
// cannot make strong assumptions about what inputs it might be called on.
impl Expr {
    pub fn eval(&self, env: &Env, ctxt: &dyn std::fmt::Debug) -> Val {
        match self {
            Expr::Var(x) => env.lookup(x).unwrap_or_else(|| panic!("failed lookup: {:?} in {:?}", x, ctxt)).clone(),
            Expr::Lit(v) => v.clone(),
            Expr::UnOp(op, e) => {
                let arg = e.eval(env, ctxt);
                match op {
                    UnOp::String2Int => {
                        if let Val::String(s) = arg {
                            Val::Int(s.parse().unwrap())
                        } else {
                            panic!("found non-string: {} as input to string2int", arg);
                        }
                    }
                }
            }
            Expr::BinOp(op, e1, e2) => {
                let lhs = e1.eval(env, ctxt);
                let rhs = e2.eval(env, ctxt);
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
