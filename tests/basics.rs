pub fn main() {
    let s = r#"A ::= 'a'; Int(n) ::= ([n > 0] ( '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9') { n := n-1 })* [n==0];"#;
    let g = ducere::parse_yakker(s).unwrap();
    dbg!(g);
}
