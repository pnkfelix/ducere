// This is actually defined at `crate::codegen::tests_for_codegen`

use thiserror::Error;
use expect_test::expect;

#[derive(Debug, Error)]
enum CodegenTestError {
    #[error("codegen io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("codegen utf8 conversion error: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),
    #[error("codegen compile fail: {0:?}")]
    CompileFail(FailedCommand),
    #[error("codegen invoke fail: {0:?}")]
    InvokeFail(FailedCommand),
}

struct FailedCommand { stdout: Vec<u8>, stderr: Vec<u8> }

impl std::fmt::Debug for FailedCommand {
    fn fmt(&self, w: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(w, r##"FailedCommand {{\n"##)?;
        write!(w, r##"    stdout: r#"{}"#\n"##, String::from_utf8_lossy(&self.stdout))?;
        write!(w, r##"    stderr: r#"{}"#\n"##, String::from_utf8_lossy(&self.stderr))?;
        write!(w, "}}")
    }
}

impl CodegenTestError {
    fn compile_fail(output: std::process::Output) -> Self {
        CodegenTestError::CompileFail(FailedCommand {
            stdout: output.stdout,
            stderr: output.stderr,
        })
    }
    fn invoke_fail(output: std::process::Output) -> Self {
        CodegenTestError::InvokeFail(FailedCommand {
            stdout: output.stdout,
            stderr: output.stderr,
        })
    }
}

#[test]
fn baseline_echo_in_rust() -> Result<(), CodegenTestError> {
    let input = r#"Hello World!"#;
    let code = r#"
fn main() -> Result<(), std::io::Error> {
    let lines = std::io::stdin().lines();
    for line in lines {
        println!("{}", line?);
    }
    Ok(())
}
"#;
    let source_temp = temp_file::with_contents(code.as_bytes());
    let input_temp = temp_file::with_contents(input.as_bytes());
    let output_temp = temp_file::empty();
    let output_path = output_temp.path().to_path_buf();
    output_temp.cleanup()?; // delete the output file
    let compile_output = std::process::Command::new("rustc")
        .arg(source_temp.path())
        .arg("-o")
        .arg(&output_path)
        .output()?;
    if !compile_output.status.success() {
        let err = CodegenTestError::compile_fail(compile_output);
        println!("err: {err}");
        return Err(err);
    }
    let invoke_output = std::process::Command::new(output_path)
        .stdin(std::fs::File::open(input_temp.path())?)
        .output()?;
    if !invoke_output.status.success() {
        let err = CodegenTestError::invoke_fail(invoke_output.clone());
        println!("err: {err}");
        return Err(err);
    }
    assert!(invoke_output.status.success());
    expect!("Hello World!\n").assert_eq(&String::from_utf8(invoke_output.stdout.clone())?);
    Ok(())
}

#[test]
fn baseline_echo_in_java() -> Result<(), CodegenTestError> {
    let input = r#"Hello World!"#;
    let code = r#"
import java.util.Scanner;
class Echo {
    public static void main(String[] args) {
        String inData;
        Scanner scan = new Scanner( System.in );

        while (scan.hasNextLine()) {
            inData = scan.nextLine();
            System.out.println(inData);
        }
    }
}
"#;
    let source_temp = temp_file::TempFile::with_suffix(".java")?
        .with_contents(code.as_bytes())?;
    let input_temp = temp_file::with_contents(input.as_bytes());
    let output_path = temp_dir::TempDir::new()?;
    let compile_output = std::process::Command::new("javac")
        .arg("-d")
        .arg(output_path.path())
        .arg(source_temp.path())
        .output()?;
    if !compile_output.status.success() {
        let err = CodegenTestError::compile_fail(compile_output);
        println!("err: {err}");
        return Err(err);
    }
    let invoke_output = std::process::Command::new("java")
        .arg("-cp")
        .arg(output_path.path())
        .arg("Echo")
        .stdin(std::fs::File::open(input_temp.path())?)
        .output()?;
    if !invoke_output.status.success() {
        let err = CodegenTestError::invoke_fail(invoke_output.clone());
        println!("err: {err}");
        return Err(err);
   }
    assert!(invoke_output.status.success());
    expect!("Hello World!\n").assert_eq(&String::from_utf8(invoke_output.stdout.clone())?);
    Ok(())
}

