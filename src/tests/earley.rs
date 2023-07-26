// This is actually defined at `crate::earley::tests_for_earley`

#![allow(unused_imports)]

use crate::earley::*;
use crate::specification_rules::earley::EarleyConfig;
use crate::tests::*;
use crate::transducer::tests_for_transducer::{fig_2_a, fig_2_b, fig_2_c};
use crate::Term;
use crate::{AbstractNode, Tree};

use expect_test::expect;

trait Fmap<X, Y> {
    type Out;
    fn fmap(&self, f: &impl Fn(&X) -> Y) -> Self::Out;
    fn fmap_into(self, f: &impl Fn(X) -> Y) -> Self::Out;
}

/*
// This is one way to do things so that you "auto-traverse" substructure; but it forces
// you to know up front, at time of impl definition, whether a given type is
// Shape or if it is Data.

impl<Y> Fmap<Tree,Y> for Tree {
    type Out = Y;
    fn fmap(&self, f: &impl Fn(&Tree) -> Y) -> Self::Out { f(self) }
    fn fmap_into(self, f: &impl Fn(Tree) -> Y) -> Self::Out { f(self) }
}

impl<X,Y,E:Fmap<X,Y>> Fmap<X,Y> for Vec<E> {
    type Out = Vec<E::Out>;
    fn fmap(&self, f: &impl Fn(&X) -> Y) -> Self::Out { self.iter().map(|e|e.fmap(f)).collect() }
    fn fmap_into(self, f: &impl Fn(X) -> Y) -> Self::Out { self.into_iter().map(|e|e.fmap_into(f)).collect() }
}

impl<X,Y,E:Fmap<X,Y>> Fmap<X,Y> for Option<E> {
    type Out = Option<E::Out>;
    fn fmap(&self, f: &impl Fn(&X) -> Y) -> Self::Out {
        match self { None => None, Some(x) => Some(x.fmap(f)) }
    }
    fn fmap_into(self, f: &impl Fn(X) -> Y) -> Self::Out {
        match self { None => None, Some(x) => Some(x.fmap_into(f)) }
    }
}
 */

impl<X,Y> Fmap<X,Y> for Vec<X> {
    type Out = Vec<Y>;
    fn fmap(&self, f: &impl Fn(&X) -> Y) -> Self::Out { self.iter().map(f).collect() }
    fn fmap_into(self, f: &impl Fn(X) -> Y) -> Self::Out { self.into_iter().map(f).collect() }
}

impl<X,Y> Fmap<X,Y> for Option<X> {
    type Out = Option<Y>;
    fn fmap(&self, f: &impl Fn(&X) -> Y) -> Self::Out {
        match self { None => None, Some(x) => Some(f(x)) }
    }
    fn fmap_into(self, f: &impl Fn(X) -> Y) -> Self::Out {
        match self { None => None, Some(x) => Some(f(x)) }
    }
}

macro_rules! assert_formatted {
    ($fmt:literal, $actual:expr, $expect:expr) => {
        {
            let actual = format!($fmt, $actual);
            $expect.assert_eq(&actual)
        }
    }
}

macro_rules! assert_dbg_eq {
    ($actual:expr, $expect:expr) => {
        assert_formatted!("{:?}", $actual, $expect)
    }
}

#[test]
fn imperative_fixed_width_integer() {
    let t = fig_2_a();
    let mut config = EarleyConfig::new_with_binding(Earley { transducer: t }, "n".into(), 1.into());
    dbg!(config.dyn_state());
    config.step(Term::C('1'));
    dbg!(config.dyn_state());

    assert_dbg_eq!(
        config.accepting(0,1),
        expect![[r#"Some([Tree(["1", {n:=0}])])"#]]);
}

#[test]
fn functional_fixed_width_integer() {
    let t = fig_2_b();
    let make_config = |x: crate::expr::Var, v: crate::expr::Val| -> EarleyConfig {
        EarleyConfig::new_with_binding(Earley { transducer: t.clone() }, x, v)
    };
    let mut config = make_config("n".into(), 1.into());
    config.step(Term::C('1'));

    assert_formatted!(
        "{:?}",
        config.accepting(0,1),
        expect![[r#"Some([Tree([dig(Tree(["1"])), int(0)(Tree([]))])])"#]]
    );

    let mut config = make_config("n".into(), 2.into());
    config.step(Term::C('9'));
    config.step(Term::C('8'));
    assert_formatted!(
        "{:?}",
        config.accepting(0,2),
        expect![[r#"Some([Tree([dig(Tree(["9"])), int(1)(Tree([dig(Tree(["8"])), int(0)(Tree([]))]))])])"#]]
    );
}
