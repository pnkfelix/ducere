#![cfg_attr(not(test), allow(dead_code))]

use crate::Tree;
use crate::expr;
use crate::transducer::{State};
use crate::Term;
use crate::earley::Earley;

use std::collections::HashMap;
use linear_map::LinearMap;

/// EarleyKey(i,j,q,E,s) is used to identify individual tree-sets in the
/// early-set-structure.
///
/// A tree T belongs to tree(i,j,q,E,s) when that tree is constructed by parsing
/// the input from position i+1 to position j, with the transducer starting in
/// state q and ending in state s, and given environment E.
///
/// FIXME: The phrasing in the paper, "Environment E is the environment that was
/// built during the course of the parse", makes it sound like E is a result
/// from the parse, but it *must* be an input due to e.g. ET-Pred, which
/// evaluates constraints; and yet ET-Bind does make E[x:=v] part of its
/// *inputs*, where [[e]]E = v ... this just doesn't seem right to me.
///
/// The right answer is probably to leave the `expr::Env` out of the *key*,
/// but keep it in the associated elements for determining lookup.
#[derive(Clone, PartialEq, Eq, Hash)]
struct EarleyKey(usize, usize, State);

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Finality { Accept, Go }

impl std::fmt::Debug for EarleyKey {
    fn fmt(&self, w: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(w, "EarleyKey({},{},{})", self.0, self.1, (self.2).0)
    }
}

// N.B. the strings in the paper are one-indexed: `w = c_1 ... c_n`; thus, the
// j-1 stuff below shifts the index into zero-indexed array (as expected in
// Rust).

#[allow(dead_code)]
#[derive(Debug)]
pub struct EarleyConfig {
    earley: Earley, // This could (and should?) be a reference.
    trees: EarleyTrees,
    step_count: usize,
    step_rule_count: usize,
    /// The number of input terminals read (and incorporated) so far.
    len: usize,
}

#[cfg(test)]
impl EarleyConfig {
    // pub(crate) fn static_state(&self) -> (&Earley,) { (&self.earley,) }
    pub(crate) fn dyn_state(&self) -> (&EarleyTrees, usize) { (&self.trees, self.len) }
}

// FIXME: does the inner indexing on key.0 end up hurting more than it helps?
// I'm currently traversing many i's to deal with it, though arguably that is
// unavoidable unless on either starts using HashMaps and/or builds up queues to
// direct future work.

/// The outer vec here is handling key.1; the second (inner) vec is key.0.
#[derive(Debug)]
pub(crate) struct EarleyTrees(Vec<Vec<EarleyMap>>);

impl EarleyTrees {

}

type CellMap<K,V> = LinearMap<K,V>;

type EarleyMap = HashMap<EarleyKey, CellMap<(expr::Env, State, Finality), Vec<Tree>>>;
type IJ = (usize, usize);

trait IndexPair {
    type T;
    fn pair_mut(&mut self, i1: usize, i2: usize) -> (&mut Self::T, &mut Self::T);
}

impl<T> IndexPair for [T] {
    type T = T;
    fn pair_mut(&mut self, i1: usize, i2: usize) -> (&mut Self::T, &mut Self::T) {
        if i1 < i2 {
            let mid = i1+1;
            let (s1, s2) = self.split_at_mut(mid);
            (&mut s1[i1], &mut s2[i2 - mid])
        } else if i1 > i2 {
            let mid = i2+1;
            let (s2, s1) = self.split_at_mut(mid);
            (&mut s1[i1-mid], &mut s2[i2])
        } else {
            assert_eq!(i1, i2);
            panic!("cannot safely take two references into same vector element at {}", i1);
        }
    }
}

impl EarleyTrees {
    fn ensure_capacity(&mut self, j:  usize) {
        assert!(j <= self.0.len(), "{j} <= len {}", self.0.len());
        if j == self.0.len() {
            let v = Vec::new();
            self.0.push(v);
        }
    }

    fn map(&self, (i, j): IJ) -> &EarleyMap {
        &self.0[j][i]
    }

    fn get_maps_for_end_index_mut(&mut self, j: usize) -> &mut Vec<EarleyMap> {
        self.ensure_capacity(j);
        &mut self.0[j]
    }

    fn get_map_mut(&mut self, (i, j): IJ) -> &mut EarleyMap {
        assert!(i <= j, "{i} <= {j}");
        #[allow(non_snake_case)]
        let trees__j = self.get_maps_for_end_index_mut(j);
        while i >= trees__j.len() {
            trees__j.push(HashMap::new());
        }
        let trees_i_j = &mut self.0[j][i];
        trees_i_j
    }

    fn get_maps_mut(&mut self, (i1, j1): IJ, (i2, j2): IJ) -> (&mut EarleyMap, &mut EarleyMap) {
        assert_ne!((i1, j1), (i2, j2));
        self.ensure_capacity(j1);
        self.ensure_capacity(j2);

        if j1 != j2 {
            let (t_j1, t_j2) = self.0.pair_mut(j1, j2);
            while t_j1.len() <= i1 { t_j1.push(Default::default()); }
            while t_j2.len() <= i2 { t_j2.push(Default::default()); }
            (&mut t_j1[i1], &mut t_j2[i2])
        } else {
            assert_eq!(j1, j2);
            assert_ne!(i1, i2);
            self.0[j1].pair_mut(i1, i2)
        }
    }
}

pub fn finality(earley: &Earley, s: State) -> Finality {
    if earley.transducer().data(s).output_if_final().is_some() {
        Finality::Accept
    } else {
        Finality::Go
    }
}

impl EarleyConfig {
    fn finality(&self, s: State) -> Finality { finality(&self.earley, s) }

    pub fn new(earley: Earley) -> Self {
        // ----------------------------------------------- ET-Init
        //  \epsilon \in tree(0, 0, q_0, [y_0 := ()], q_0)
        Self::new_with_binding(earley, expr::y_0(), ().into())
    }

    pub fn new_with_binding(earley: Earley, x: expr::Var, v: expr::Val) -> Self {
        let q_0 = earley.transducer().start_state();
        let env = expr::Env::bind(x, v);

        let env_trees: CellMap<_, _> = vec![(((env, q_0, finality(&earley, q_0))), vec![Tree(vec![])])].into_iter().collect();
        let mut trees_0_0 = HashMap::new();
        trees_0_0.insert(EarleyKey(0, 0, q_0), env_trees);
        let trees = EarleyTrees(vec![vec![trees_0_0]]);
        EarleyConfig { earley, trees, len: 0, step_count: 0, step_rule_count: 0 }
    }

    pub fn step(&mut self, t: Term) {
        let len = self.len;
        self.step_count += 1;
        loop {
            let changed = self.apply_rules(t.clone());
            if let Changed::Unchanged = changed {
                break;
            }
        }

        assert_eq!(len + 1, self.len);
    }

    fn apply_rules(&mut self, t: Term) -> Changed {
        self.step_rule_count += 1;
        dbg!((self.step_count, self.step_rule_count));
        // FIXME would overloading `?` be a good thing here?
        if let Changed::Changed = dbg!(self.apply_et_pred()) { return Changed::Changed; }
        if let Changed::Changed = dbg!(self.apply_et_bind()) { return Changed::Changed; }
        // if let Changed::Changed = self.apply_et_phi() { return Changed::Changed; }
        if let Changed::Changed = dbg!(self.apply_et_call()) { return Changed::Changed; }
        if let Changed::Changed = dbg!(self.apply_et_return()) { return Changed::Changed; }

        // The order here *might* matter. In particular; if this `if let` came
        // first in the series, then I am not sure if that might make us skip
        // certain paths through the state machine. (But also, this rules
        // application strategy is itself worrisome; it seems like it could get
        // stuck applying the same rule over and over if the transducer has a
        // self-loop.)
        if let Changed::Changed = dbg!(self.apply_et_term(t)) { self.len += 1; return Changed::Changed; }

        return Changed::Unchanged;
    }

    // FIXME: in all the for i in 0..k loops below: we shouldn't need to loop
    // over all these i's, right?
    //
    // At the very least we should be able to all cases of interest at the
    // time when the tree T is inserted into tree(i, j-1, q, E, r), based
    // on the presence/absence of a transducer arc r \to^{c_j} s.

    ///  T \in tree(i, j-1, q, E, r)
    ///  r \to^{c_j} s
    /// ------------------------------- ET-Term
    ///  T c_j \in tree(i, j, q, E, s)
    fn apply_et_term(&mut self, t: Term) -> Changed {
        let mut changed = Changed::Unchanged;

        // the next character is at len+1, which equals j because j is one-indexed.
        let j = self.len + 1;

        for i in 0..j {
            let (tree_i_jsub1, tree_i_j) = self.trees.get_maps_mut((i, j-1), (i, j));
            for (&EarleyKey(_, _, q), ref env_trees) in tree_i_jsub1 {
                for (&(ref env, r, _), trees) in env_trees.iter() {
                    // FIXME StateData API should let one do action-filtered lookup
                    for &s in self.earley.transducer().data(r).term_transitions(&t) {
                        let f = finality(&self.earley, s);
                        for tree in trees.iter() {
                            let mut tree = tree.clone();
                            tree.extend_term(t.clone());
                            let key = EarleyKey(i, j, q);
                            let env_trees = tree_i_j
                                .entry(key.clone())
                                .or_insert(CellMap::new());
                            let env_state = (env.clone(), s, f);
                            env_trees.entry(env_state.clone()).or_insert(vec![]);
                            let cell = env_trees.get_mut(&env_state).unwrap();
                            if !cell.contains(&tree) {
                                cell.push(tree);
                                dbg!(key);
                                changed = Changed::Changed;
                            }
                        }
                    }
                }
            }
        }

        return changed;
    }

    //  T \in tree(i, j, q, E, r)
    //  r \to^{e} s
    //  [[ e ]] E == true
    // --------------------------- ET-Pred
    //  T \in tree(i, j, q, E, s)
    fn apply_et_pred(&mut self) -> Changed {
        let j = self.len;

        let mut changed = Changed::Unchanged;

        for i in 0..=j {
            let mut to_add: Vec<(EarleyKey, (expr::Env, State, Finality), Vec<Tree>)> = Vec::new();
            let tree_i_j = self.trees.get_map_mut((i, j));
            for (&EarleyKey(_, _, q), env_trees) in &*tree_i_j {
                for (&(ref env, r, _), trees) in env_trees.iter() {
                    for (e, &s) in self.earley.transducer().data(r).pred_transitions() {
                        if let expr::Val::Bool(true) = e.eval(env) {
                            let f = finality(&self.earley, s);
                            to_add.push((EarleyKey(i, j, q),
                                         ((*env).clone(), s, f),
                                         trees.clone()));
                        }
                    }
                }
            }

            for (key, env, trees_to_add) in to_add {
                let trees = tree_i_j.entry(key).or_insert(CellMap::new());
                trees.entry(env.clone()).or_insert(vec![]);
                for tree in trees_to_add {
                    let cell = trees.get_mut(&env).unwrap();
                    if !cell.contains(&tree) {
                        cell.push(tree);
                        changed = Changed::Changed;
                    }
                }
            }
        }

        return changed;
    }

    //  T \in tree(i, j, q, E, r)
    //  r \to^{x:=e} s
    //  [[ e ]] E == v
    //  x != y_0
    // -------------------------------------- ET-Bind
    //  T{x:=v} \in tree(i, j, q, E[x:=v], s)
    fn apply_et_bind(&mut self) -> Changed {
        let j = self.len;

        let mut changed = Changed::Unchanged;

        for i in 0..=j {
            // dbg!(i);
            let mut to_add: Vec<(EarleyKey, (expr::Env, State, Finality), Tree)> = Vec::new();
            let tree_i_j = self.trees.get_map_mut((i, j));
            // dbg!(&tree_i_j);
            for (&EarleyKey(_, _, q), ref env_trees) in &*tree_i_j {
                // dbg!((i,q,r));
                for (&(ref env, r, _), trees) in env_trees.iter() {
                    // dbg!((i,q,r,env));
                    for ((x, e), &s) in self.earley.transducer().data(r).bind_transitions() {
                        // dbg!((i,q,r,env,x,e,s));
                        let v = e.eval(env);
                        let mut env = (*env).clone();
                        env.extend(x.clone(), v.clone());
                        for t in trees {
                            // dbg!((i,q,r,&env,x,e,s,t));
                            let mut t = t.clone();
                            t.extend_bind(x.clone(), v.clone());
                            let f = finality(&self.earley, s);
                            to_add.push((EarleyKey(i, j, q),
                                         (env.clone(), s, f),
                                         t));
                        }
                    }
                }
            }
            for (key, env_state, tree) in to_add {
                let trees = tree_i_j.entry(key).or_insert(CellMap::new());
                trees.entry(env_state.clone()).or_insert(vec![]);
                let cell = trees.get_mut(&env_state).unwrap();
                if !cell.contains(&tree) {
                    cell.push(tree);
                    changed = Changed::Changed;
                }
            }
        }

        changed
    }

    #[cfg(not_yet)]
    //  T \in tree(i, k-1, q, E, r)
    //  r \to^{\phi(e)} s
    //  c_k .. c_j \in \Phi(\phi)(v)
    //  [[ e ]] E == v
    // --------------------------------------- ET-\phi
    //  T<c_k .. c_j> \in tree(i, j, q, E, s)
    fn apply_et_phi(&mut self) -> Changed {
        let j = self.len;

        let mut changed = Changed::Unchanged;

        unimplemented!()
    }

    //  T \in tree(i, j, q, E, r)
    //  r \to^{call(e)} s
    //  [[ e ]] E == v
    // ------------------------------------------- ET-Call
    //  \epsilon \in tree(j, j, s, [y_0 := v], s)
    fn apply_et_call(&mut self) -> Changed {
        let j = self.len;

        let mut changed = Changed::Unchanged;

        for i in 0..=j {
            let mut to_add = Vec::new();
            let tree_i_j = self.trees.get_map_mut((i, j));
            for (&EarleyKey(_, _, _q), env_trees) in &*tree_i_j {
                for (&(ref env, r, _), trees) in env_trees.iter() {
                    if trees.len() == 0 { continue; }
                    for &(ref e, s) in self.earley.transducer().data(r).calls() {
                        let v = e.eval(env);
                        let new_env = expr::Env::bind(expr::y_0(), v);
                        let f = finality(&self.earley, s);
                        let key = EarleyKey(j, j, s);
                        to_add.push((key, (new_env, s, f), Tree(vec![])));
                    }
                }
            }

            let tree_j_j = self.trees.get_map_mut((j, j));
            for (key, env_state, tree) in to_add {
                let trees = tree_j_j.entry(key).or_insert(CellMap::new());
                trees.entry(env_state.clone()).or_insert(vec![]);
                let cell = trees.get_mut(&env_state).unwrap();
                if !cell.contains(&tree) {
                    cell.push(tree);
                    changed = Changed::Changed;
                }
            }
        }

        changed
    }

    //  [[ e_1 ]] E_2 == [[ e_2 ]] E_2 == E_1(y_0) == v
    //  x != y_0
    //  T_1 \in tree(k, j, q, E_1, r)
    //  t \to^{call(e_1)} q
    //  r \mapsto A
    //  T_2 \in tree(i, j, s, E_2, t)
    //  t \to^{x:=A(e_2)} u
    // ------------------------------------------------- ET-Return
    //  T_2 x:A(v)<T_1> \in
    //          tree(i, j, s, E_2[x:=leaves(T_1)], u)
    fn apply_et_return(&mut self) -> Changed {
        let j = self.len;
        let transducer = self.earley.transducer();

        let mut changed = Changed::Unchanged;

        for i in 0..=j {
            let tree_i_j = self.trees.map((i, j));
            let mut to_add = Vec::new();

            for k in 0..=j {
                let tree_k_j = self.trees.map((k, j));
                for (&EarleyKey(_, _, s), env_trees_2) in tree_i_j {
                    for (&(ref env_2, t, _), trees_2) in env_trees_2.iter() {
                        for &(ref e_1, q) in transducer.data(t).calls() {
                            for (&EarleyKey(_, _, q_,), env_trees_1) in tree_k_j {
                                for (&(ref env_1, r, _), trees_1) in env_trees_1.iter() {
                                    // FIXME: there should be a better way to zero in on
                                    // specific cases of `q` as start state and `r` is a
                                    // accepted state.
                                    if q != q_ { continue; }
                                    let non_terms = if let Some(non_terms) = transducer.data(r).output_if_final() {
                                        non_terms
                                    } else {
                                        continue;
                                    };
                                    assert_eq!(q, q_);
                                    for ((opt_x, nt, opt_e), &u) in transducer.data(t).nonterm_transitions(non_terms) {
                                        let opt_x = opt_x.clone();
                                        let nt = nt.clone();
                                        let unit: expr::Expr = ().into();
                                        let e_2 = if let Some(e) = opt_e { e } else { &unit };
                                        let v = {
                                            let v_0 = expr::Expr::Var(expr::y_0()).eval(env_1);
                                            let v_1 = e_1.eval(env_2);
                                            let v_2 = e_2.eval(env_2);
                                            if v_0 == v_1 && v_1 == v_2 {
                                                v_0
                                            } else {
                                                continue;
                                            }
                                        };
                                        for t_2 in trees_2 {
                                            for t_1 in trees_1 {
                                                let mut t_2 = t_2.clone();
                                                t_2.extend_parsed(opt_x.clone(), nt.clone(), v.clone(), t_1.clone());
                                                let mut env_2 = (*env_2).clone();
                                                if let Some(x) = opt_x.clone() {
                                                    env_2.extend(x, expr::Val::String(t_1.leaves().iter().map(|x|x.as_ref()).collect()));
                                                }
                                                let f = self.finality(u);
                                                let key = EarleyKey(i, j, s);
                                                to_add.push((key, (env_2.clone(), u, f), t_2));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            for (key, env_2, t_2) in to_add {
                let tree_i_j = self.trees.get_map_mut((i, j));
                let trees = tree_i_j.entry(key).or_insert(CellMap::new());
                trees.entry(env_2.clone()).or_insert(vec![]);
                let cell = trees.get_mut(&env_2).unwrap();
                if !cell.contains(&t_2) {
                    cell.push(t_2);
                    changed = Changed::Changed;
                }
            }
        }

        changed
    }
}

#[derive(PartialEq, Eq, Debug)]
enum Changed { Unchanged, Changed }


