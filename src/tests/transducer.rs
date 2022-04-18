// This is actually defined at `crate::transducer::tests_for_transducer`

use crate::transducer::*;
use crate::tests::*;
use crate::expr;
use crate::specification_rules::transducer::TransducerConfig;
use crate::yakker;

fn expr(s: &str) -> expr::Expr {
    yakker::ExprParser::new().parse(s).unwrap()
}

pub fn fig_2_a() -> Transducer {
    let s0 = State(0);
    let s1 = State(1);
    let s2 = State(2);
    let s3 = State(3);
    let d0 = StateBuilder::final_state(format!("int")).build();
    let d1 = StateBuilder::labelled("1".into())
        .constraint(expr("n == 0"), s0)
        .constraint(expr("n > 0"), s2)
        .build();
    let d2 = StateBuilder::labelled("2".into())
        .term_range(('0', '9'), s3)
        .build();
    let d3 = StateBuilder::labelled("3".into())
        .binding(Var("n".into()), expr("n-1"), s1)
        .build();
    Transducer {
        states: vec![(s0, d0), (s1, d1), (s2, d2), (s3, d3)].into_iter().collect()
    }
}

#[test]
fn imperative_fixed_width_integer() {
    let t = fig_2_a();
    let one_digit = expr::Env::bind("n".into(), 1.into());
    let config = TransducerConfig::fresh(State(1), one_digit);
    let next = config.clone().goto(State(2));
    assert_eq!(t.matches(config.clone(), &input("1")), vec![(next.clone(), 0)]);
    let config = next;
    let next = config.clone().goto(State(3)).term(Term::C('1'));
    assert_eq!(t.matches(config.clone(), &input("1")), vec![(next.clone(), 1)]);
    let config = next;
    let next = config.clone().goto(State(1)).bind("n".into(), 0.into());
    assert_eq!(t.matches(config.clone(), &input("")), vec![(next.clone(), 0)]);
    let config = next;
    let next = config.clone().goto(State(0));
    assert_eq!(t.matches(config.clone(), &input("")), vec![(next.clone(), 0)]);
    assert_eq!(t.data(next.state()).output_if_final(), Some(["int".into()].as_slice()));
}

pub fn fig_2_b() -> Transducer {
    let s0 = State(0);
    let s1 = State(1);
    let s2 = State(2);
    let s3 = State(3);
    let s4 = State(4);
    let s5 = State(5);
    let s6 = State(6);
    let d0 = StateBuilder::final_state(format!("int")).build();
    let d1 = StateBuilder::parameterized("1".into(), "n".into())
        .constraint(expr("n == 0"), s0)
        .constraint(expr("n > 0"), s2)
        .build();
    let d2 = StateBuilder::labelled("2".into())
        .call(Expr::Lit(().into()), s5)
        .non_term(NonTerm("dig".into()), s3)
        .build();
    let d3 = StateBuilder::labelled("3".into())
        .call(expr("n - 1"), s1)
        .non_term((NonTerm("int".into()), expr("n - 1")), s4)
        .build();
    let d4 = StateBuilder::final_state(format!("int"))
        .build();
    let d5 = StateBuilder::labelled("5".into())
        .term_range(('0', '9'), s6)
        .build();
    let d6 = StateBuilder::final_state(format!("dig"))
        .build();
    Transducer {
        states: vec![(s0, d0), (s1, d1), (s2, d2), (s3, d3), (s4, d4), (s5, d5), (s6, d6)].into_iter().collect()
    }
}

#[test]
fn functional_fixed_width_integer() {
    let t = fig_2_b();
    let one_digit = expr::Env::bind("n".into(), 1.into());
    let config = TransducerConfig::fresh(State(1), one_digit);
    let next = config.clone().goto(State(2));
    assert_eq!(t.matches(config.clone(), &input("1")), vec![(next.clone(), 0)]);
    let config = next;
    let next = config.clone().call(State(5), None);
    assert_eq!(t.matches(config.clone(), &input("1")), vec![(next.clone(), 0)]);
    let config = next;
    let next = config.clone().goto(State(6)).term('1'.into());
    assert_eq!(t.matches(config.clone(), &input("1")), vec![(next.clone(), 1)]);
    let config = next;
    let next = config.clone().return_to(None, "1".into(), "dig".into(), ().into(), State(3));
    assert_eq!(t.matches(config.clone(), &input("")), vec![(next.clone(), 0)]);
    let config = next;
    let next = config.clone().call(State(1), Some(("n".into(), &expr("n-1"), 0.into())));
    assert_eq!(t.matches(config.clone(), &input("")), vec![(next.clone(), 0)]);
    let config = next;
    let next = config.clone().goto(State(0));
    assert_eq!(t.matches(config.clone(), &input("")), vec![(next.clone(), 0)]);
    assert_eq!(t.data(next.state()).output_if_final(), Some(["int".into()].as_slice()));
    let config = next;
    let next = config.clone().return_to(None, "".into(), "int".into(), 0.into(), State(4));
    assert_eq!(t.matches(config.clone(), &input("")), vec![(next.clone(), 0)]);
    let config = next;
    assert_eq!(t.matches(config.clone(), &input("")), vec![]);
    assert_eq!(config.peek().tree().leaves().into_iter().collect::<String>(), "1".to_string());
}
