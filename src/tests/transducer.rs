// This is actually defined at `crate::transducer::tests_for_transducer`

use crate::transducer::*;
use crate::tests::*;
use crate::expr;
use crate::specification_rules::transducer::TransducerConfig;
use crate::yakker;
use crate::toyman;

fn expr(s: &str) -> expr::Expr {
    let lex = toyman::Lexer::new(s);
    yakker::ExprParser::new().parse(s, lex).unwrap()
}

const fn state_iota<const DIM: usize>() -> [State; DIM] {
    let mut array = [State(0); DIM];
    let mut i = 0;
    while i < DIM {
        array[i] = State(i);
        i += 1;
    }
    array
}

pub fn fig_2_a() -> Transducer {
    let s: [State; 4] = state_iota();
    let d0 = StateBuilder::final_state(format!("int")).build();
    let d1 = StateBuilder::labelled("1".into())
        .constraint(expr("n == 0"), s[0])
        .constraint(expr("n > 0"), s[2])
        .build();
    let d2 = StateBuilder::labelled("2".into())
        .term_range(('0', '9'), s[3])
        .build();
    let d3 = StateBuilder::labelled("3".into())
        .binding(Var("n".into()), expr("n-1"), s[1])
        .build();
    Transducer {
        states: vec![(s[0], d0), (s[1], d1), (s[2], d2), (s[3], d3)].into_iter().collect()
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
    let s: [State; 7] = state_iota();
    let d0 = StateBuilder::final_state(format!("int")).build();
    let d1 = StateBuilder::parameterized("1".into(), "n".into())
        .constraint(expr("n == 0"), s[0])
        .constraint(expr("n > 0"), s[2])
        .build();
    let d2 = StateBuilder::labelled("2".into())
        .call(Expr::Lit(().into()), s[5])
        .non_term(NonTerm("dig".into()), s[3])
        .build();
    let d3 = StateBuilder::labelled("3".into())
        .call(expr("n - 1"), s[1])
        .non_term((NonTerm("int".into()), expr("n - 1")), s[4])
        .build();
    let d4 = StateBuilder::final_state(format!("int"))
        .build();
    let d5 = StateBuilder::labelled("5".into())
        .term_range(('0', '9'), s[6])
        .build();
    let d6 = StateBuilder::final_state(format!("dig"))
        .build();
    Transducer {
        states: vec![(s[0], d0), (s[1], d1), (s[2], d2), (s[3], d3),
                     (s[4], d4), (s[5], d5), (s[6], d6)].into_iter().collect()
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

pub fn fig_2_c() -> Transducer {
    let s: [State; 11] = state_iota();
    let d0 = StateBuilder::final_state("A".into()).build();
    let d1 = StateBuilder::labelled("1".into())
        .non_term((NonTerm("B".into()), expr("()")), s[2])
        .non_term((NonTerm("C".into()), expr("()")), s[3])
        .call(expr("()"), s[4])
        .build();
    let d2 = StateBuilder::labelled("2".into()).term('?', s[0]).build();
    let d3 = StateBuilder::labelled("3".into()).term('1', s[8]).build();
    let d4 = StateBuilder::labelled("4".into()).term('x', s[5]).build();
    let d5 = StateBuilder::labelled("5".into()).term('-', s[6]).term('+', s[7]).build();
    let d6 = StateBuilder::labelled("6".into()).term('x', s[9]).build();
    let d7 = StateBuilder::labelled("7".into()).term('x', s[10]).build();
    let d8 = StateBuilder::final_state_labelled_and_accepting("A8".into(), vec!["A".to_string()]).build();
    let d9 = StateBuilder::final_state_labelled_and_accepting("C".into(), vec!["C".to_string()]).build();
    let d10 = StateBuilder::final_state_labelled_and_accepting("B,C".into(), vec!["B".to_string(), "C".to_string()]).build();
    Transducer {
        states: vec![(s[0], d0), (s[1], d1), (s[2], d2), (s[3], d3),
                     (s[4], d4), (s[5], d5), (s[6], d6), (s[7], d7),
                     (s[8], d8), (s[9], d9), (s[10], d10)].into_iter().collect(),
    }
}

#[test]
fn left_factoring() {
    let t = fig_2_c();
    let config = TransducerConfig::fresh(State(1), expr::Env::empty());
    let next = config.clone().call(State(4), None);
    assert_eq!(t.matches(config.clone(), &input("x-x")), vec![(next.clone(), 0)]);
}
