use crate::{expr, Bother, Grammar, RegularRightSide, Rule, Term};

use crate::rendering::Rendered;

struct Cross<I: Iterator, ToJ, J> where I: Iterator,
{
    i_iter: I,
    to_j: ToJ,
    cursor: Option<(I::Item, J)>,
}

impl<I, ToJ, J>  Cross<I, ToJ, J> where
    I: Iterator, ToJ: for<'a> FnMut(&'a I::Item) -> J, J: Iterator
{
    fn new(mut i_iter: I, mut to_j: ToJ) -> Self {
        let curr_i = i_iter.next();
        let cursor = curr_i.map(|i| {
            let js = to_j(&i);
            (i, js)
        });
        Self { i_iter, to_j, cursor }
    }
}

impl<I, ToJ, J> Iterator for Cross<I, ToJ, J> where
    I: Iterator, I::Item: Clone, ToJ: for<'a> FnMut(&'a I::Item) -> J, J: Iterator
{
    type Item = (I::Item, J::Item);
    fn next(&mut self) -> Option<(I::Item, J::Item)> {
        let (ref mut i, ref mut j_iter) = if let Some(t) = self.cursor.as_mut() { t } else { return None; };
        'advance_j: loop {
            if let Some(j) = j_iter.next() {
                return Some((i.clone(), j));
            }
            // else: curr_j_iter exhausted. Advance to next i and try that.
            let new_i = match self.i_iter.next() {
                None => { self.cursor = None; return None; }
                Some(i) => { i }
            };
            let new_j_iter = (self.to_j)(&new_i);
            *i = new_i;
            *j_iter = new_j_iter;
            continue 'advance_j;
        }
    }
}


impl Grammar {
    // This implements the relation
    // env ⊢ w ∈ r ⇒ Env
    //
    // Notes:
    //
    //  * the given `w` has to *exactly* match the needs of the rule.
    //
    //  * there may be multiple ways for given `w` to match a rule, and that
    //  means different resulting env. The Yakker paper doesn't say much about
    //  this; my reading of the theorems, especially of Earley soundness, that
    //  if different derivations exist, then you don't know which one you will
    //  get. (After all, if you didn't allow for that, and forced the system to
    //  yield every derivation, then that would preclude most optimizations for
    //  the parser.)
    //     * revisiting this point: it is possible cannot avoid searching through
    //       multiple potential matches. One issue is constraints [PRED]: if you
    //       can get distinct environments out, you need to find the ones that will
    //       satisfy PRED.
    pub fn matches<'s>(&'s self, w: &'s [Term], r: &'s RegularRightSide) -> Box<dyn Iterator<Item=expr::Env> + 's> {
        self.matches_recur(expr::Env::empty(), w, r, 0).0
    }

    fn split_matches<'s, 'a>(&'s self, mut splits: impl Clone + Iterator<Item=usize> + 's, env: expr::Env, w: &'s [Term], mut rs: impl Clone + Iterator<Item=&'s RegularRightSide> + 's, depth: usize) -> (Box<dyn Iterator<Item=expr::Env> + 's>, usize) {
        let (split, r) = match (splits.next(), rs.next()) {
            (Some(split), Some(r)) => (split, r),
            _ => {
                assert_eq!(w.len(), 0);
                return (Some(expr::Env::empty()).b_iter(), depth);
            }
        };

        let (w_pre, w_post) = w.split_at(split);
        let (e_pre_iter, accum) = self.matches_recur(env.clone(), w_pre, &r, depth+1);
        (Box::new(e_pre_iter.flat_map(move |left_env| {
            let left_env = left_env.clone();
            self.split_matches(splits.clone(), env.clone().concat(left_env.clone()), w_post, rs.clone(), accum+1).0
                .map(move |env| left_env.clone().concat(env))
        })), accum)
    }

    fn matches_recur<'s>(&'s self, env: expr::Env, w: &'s [Term], r: &'s RegularRightSide, depth: usize) -> (Box<dyn Iterator<Item=expr::Env> + 's>, usize) {
        let accum = depth;
        let indent: String = std::iter::repeat(' ').take(depth).collect();
        println!("{}env: {} w: {:?} r: `{}`", indent, env, w.rendered(), r);

        match r {
            // GL-EPS
            RegularRightSide::EmptyString =>
                if w.len() == 0 {
                    (Some(expr::Env::empty()).b_iter(), accum)
                } else {
                    (None.b_iter(), accum)
                },
            // GL-TERM
            RegularRightSide::Term(t) =>
                if t.matches(w) {
                    (Some(expr::Env::empty()).b_iter(), accum)
                } else {
                    (None.b_iter(), accum)
                },
            // GL-PRED
            RegularRightSide::Constraint(e) =>
                if e.eval(&env, &(file!(), line!())) == expr::TRUE {
                    (Some(expr::Env::empty()).b_iter(), accum)
                } else {
                    (None.b_iter(), accum)
                }
            // GL-BIND
            RegularRightSide::Binding { x, e } =>
                if w.len() == 0 {
                    (Some(expr::Env::bind(x.clone(), e.eval(&env, &(file!(), line!())))).b_iter(), accum)
                } else {
                    (None.b_iter(), accum)
                },
            // GL-φ
            RegularRightSide::Blackbox(bb, e) => {
                let v = e.eval(&env, &(file!(), line!()));
                let phi = (bb.from_val)(v);
                // The BB interface is iterator based; but the spec for GL-φ
                // implies that the whole string must be matched. We ensure this
                // by checking that the iterator was exhausted by the blackbox.
                //
                // (This is clearly a suboptimal interface overall; i.e. this
                // system will potentially invoke the same blackbox repeatedly,
                // when what we *should* do is have a blackbox interface that
                // allows it to incrementally process more of the input on
                // demand, and signal recognition on successive prefixes.
                //
                // In any case, in general the Iterator spec does not guarantee
                // that successive calls to `next()` will return `None`; but
                // calling `fuse()` *will* ensure this.
                let mut cs = w.iter().fuse();
                let accepted = phi.accept(&mut cs);
                if accepted.is_some() && cs.next().is_none() {
                    (Some(expr::Env::empty()).b_iter(), accum)
                } else {
                    (None.b_iter(), accum)
                }
            }
            // GL-A
            RegularRightSide::NonTerm { x, A, e } => {
                let Rule { label: _ , lhs: _a, param: opt_var, rhs: subrule } =
                    self.rule(A).unwrap();
                let subenv = match (e, opt_var) {
                    (None, None) =>
                        // great: non-parameterized terminals don't need values
                        expr::Env::empty(),
                    (Some(e), Some(y_0)) => {
                        let v = e.eval(&env, &(file!(), line!()));
                        expr::Env::bind(y_0.clone(), v)
                    }
                    (Some(e), None) => {
                        // not as great: I'd prefer to not use y_0 formalism.
                        let v = e.eval(&env, &(file!(), line!()));
                        expr::Env::bind(expr::y_0(), v)
                    }
                    (None, Some(y_0)) => {
                        panic!("provided expr argument {:?} \
                                to *unparameterized* non-term {:?}({:?})",
                               e, _a, y_0);
                    }
                };
                let (subresults, accum) = self.matches_recur(subenv, w, subrule, depth+1);
                let x = x.clone();
                (Box::new(subresults.map(move |_discarded_env| {
                    match &x {
                        Some(x) => expr::Env::bind(x.clone(), w.into()),
                        None => expr::Env::empty(),
                    }
                })), accum)
            }
            // GL-SEQ
            //
            // this is where we *really* see the weakness of the API
            // being used here: instead of streaming through the input
            // and having some way to save intermediate results,
            // this code is forced to try each partition of the strings,
            // (including empty strings!).
            RegularRightSide::Concat(r1, r2) => {
                for i in 0..=w.len() {
                    let wr = w.rendered();
                    let (w1,  w2) = w.split_at(i);
                    println!("{}`{:?}` in Concat(`{}`,`{}`) trial i={} yields {:?} {:?}",
                             indent, wr, r1, r2, i, w1.rendered(), w2.rendered());
                    let (subresults_1, accum) = self.matches_recur(env.clone(), w1, r1, depth+1);
                    let cross = Cross::new(
                        subresults_1,
                        |env1| {
                            let new_env = env.clone().concat(env1.clone());
                            let (subresults_2, accum) = self.matches_recur(new_env, w2, r2, accum+1);
                            // println!("{}i: {} env2: {} w2: {:?} r2: `{}`, w2_in_r2: {:?}", indent, i, new_env, w2.rendered(), r2, format!("{}", &env2));
                            subresults_2.map(move |sr| (sr, accum))
                        });
                    for (env1, (env2, accum)) in cross {
                        println!("{}i: {} env: {} w1: {:?} r1: `{}` env1: {} w2: {:?} r2: `{}` env2: {}",
                                 indent, i, env,
                                 w1.rendered(), r1, env1.rendered(),
                                 w2.rendered(), r2, env2.rendered(),
                        );
                        return (Some(env1.concat(env2)).b_iter(), accum);
                    }
                }
                return (None.b_iter(), accum);
            }
            // GL-ALTL + GL-ALTR
            RegularRightSide::Either(r1, r2) => {
                let (result1, accum) = self.matches_recur(env.clone(), w, r1, depth+1);
                let (result2, accum) = self.matches_recur(env, w, r2, accum+1);
                return (Box::new(result1.chain(result2)), accum);
            }
            // GL-*
            //
            // Remember that note up above about seeing the weakness of the API
            // on GL-SEQ, because it needs to just guess the split point to use?
            // This is in theory worse, since this rule as stated is
            // parameterized over any choice of k splits of the string, where
            // *empty strings are permitted*. and where the environment is being
            // built up as we go.
            //
            // But, in practice, we need to keep in mind that the whole goal is
            // to process the input string. I haven't yet figured out whether
            // the constraint language could force one to match an arbitrary
            // number of empty strings, but I'm not going to try to handle that
            // case here for now. Instead, I'm going to assume that for Kleene
            // closure that every substring of a non-empty input *is* non-empty.
            RegularRightSide::Kleene(sr) => {
                // if the string is empty, we trivially match
                if w.len() == 0 {
                    println!("{}Kleene trivial empty match env: {} w: {:?} r: `{}`",
                             indent, env, w.rendered(), r);
                    return (Some(expr::Env::empty()).b_iter(), accum);
                }

                let r = sr;
                // otherwise, we will enumerate the ways to break up the string.
                // This is admittedly very dumb (e.g. it will mean we repeatedly
                // check the same prefix an absurd number of times), but its a
                // place to start.
                //
                // (One way to make this a little bit smarter would be to fold
                // in the parts generation code, and then figure out how to
                // "skip ahead" when we know a prefix of a given length doesn't
                // match and thus will *never* match.)
                let num_parts_iter = 1..=w.len();
                let indent = indent.clone();
                (Box::new(num_parts_iter.flat_map(move |num_parts| {
                    let env = env.clone();
                    let indent = indent.clone();
                    parts(w.len(), num_parts).flat_map(move |splits| {
                        println!("{}Kleene considering splits {:?}", indent, splits);
                        let all_r = std::iter::repeat(&**r);
                        self.split_matches(splits.into_iter(), env.clone(), w, all_r, accum+1).0
                    })
                })), accum)
            }
            RegularRightSide::EmptyLanguage => (None.b_iter(), accum),
        }
    }
}

#[test]
fn test_parts() {
    assert_eq!(parts(3, 1).collect::<Vec<_>>(),
               vec![vec![3]]);
    assert_eq!(parts(3, 2).collect::<Vec<_>>(),
               vec![vec![1,2], vec![2,1]]);
    assert_eq!(parts(4, 2).collect::<Vec<_>>(),
               vec![vec![1,3], vec![2,2], vec![3,1]]);
    assert_eq!(parts(4, 3).collect::<Vec<_>>(),
               vec![vec![1,1,2], vec![1,2,1],
                    vec![2,1,1]]);
    assert_eq!(parts(5, 3).collect::<Vec<_>>(),
               vec![vec![1,1,3], vec![1,2,2], vec![1,3,1],
                    vec![2,1,2], vec![2,2,1],
                    vec![3,1,1]]);
}

// given positive target T and positive count N where N <= T, returns iterator
// over length-N vectors of positive numbers that add up to T.
fn parts(target: usize, count: usize) -> impl Iterator<Item=Vec<usize>> {
    assert!(count <= target);

    // base case: trivial count
    if count == 1 {
        return vec![vec![target]].into_iter();
    }

    // recursive case: take a number, recur, map over results. repeat.
    let mut results: Vec<Vec<usize>> = Vec::new();
    for take in 1..=(target - count + 1) {
        let sub_target = target - take;
        let sub_count = count - 1;
        for sub_part in parts(sub_target, sub_count) {
            let mut took = vec![take];
            took.extend(sub_part.into_iter());
            results.push(took);
        }
    }
    return results.into_iter();
}
