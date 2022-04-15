// This is actually defined at `crate::transducer::tests_for_transducer`

use crate::transducer::*;
use crate::tests::*;
use crate::expr;
use crate::specification_rules::transducer::{TransducerConfig};
use crate::yakker;

pub fn fig_2_a() -> Transducer {
    let s0 = State(0);
    let s1 = State(1);
    let s2 = State(2);
    let s3 = State(3);
    let d0 = StateBuilder::final_state(format!("int")).build();
    let d1 = StateBuilder::labelled("1".into())
        .constraint(yakker::ExprParser::new().parse("n == 0").unwrap(), s0)
        .constraint(yakker::ExprParser::new().parse("n > 0").unwrap(), s2)
        .build();
    let d2 = StateBuilder::labelled("2".into())
        .term(Term::C('0'), s3)
        .term(Term::C('1'), s3)
        .term(Term::C('2'), s3)
        .term(Term::C('3'), s3)
        .term(Term::C('4'), s3)
        .term(Term::C('5'), s3)
        .term(Term::C('6'), s3)
        .term(Term::C('7'), s3)
        .term(Term::C('8'), s3)
        .term(Term::C('9'), s3)
        .build();
    let d3 = StateBuilder::labelled("3".into())
        .binding(Var("n".into()), yakker::ExprParser::new().parse("n-1").unwrap(), s1)
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
    let next = config.clone().map_tip(|frame|frame.with_curr(State(2)));
    assert_eq!(t.matches(config.clone(), &input("1")), vec![(next.clone(), 0)]);
    let config = next;
    let next = config.clone().map_tip(|frame|frame.with_curr(State(3)).with_term(Term::C('1')));
    assert_eq!(t.matches(config.clone(), &input("1")), vec![(next.clone(), 1)]);
    let config = next;
    let next = config.clone().map_tip(|frame|frame.with_curr(State(1)).with_bind("n".into(), 0.into()));
    assert_eq!(t.matches(config.clone(), &input("")), vec![(next.clone(), 0)]);
    let config = next;
    let next = config.clone().map_tip(|frame|frame.with_curr(State(0)));
    assert_eq!(t.matches(config.clone(), &input("")), vec![(next.clone(), 0)]);
    assert_eq!(t.data(next.state()).output_if_final(), Some(["int".into()].as_slice()));
}