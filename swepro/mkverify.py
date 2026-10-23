import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
old = """    5. Test edge cases to ensure your fix is robust
    6. Submit your changes and finish your work by issuing the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`."""
new = """    5. Test edge cases to ensure your fix is robust
    6. Before submitting, verify against the task statement:
       a. Re-read the task statement above and write a numbered checklist of EVERY requirement it states: each new function,
          class, option, parameter, return value, exact message text, error that must be raised, and type signature.
       b. For each item, write or run a test that checks exactly that behaviour; fix anything that fails.
       c. Run the project's full existing test suite and its type checker or compiler (for example tsc --noEmit, go vet, mypy)
          on all changed files AND on test files, and fix every error you introduced.
       d. Only when every checklist item is verified, submit.
    7. Submit your changes and finish your work by issuing the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`."""
assert s.count(old) == 1, s.count(old)
open(dst, "w").write(s.replace(old, new))
print("ok")
