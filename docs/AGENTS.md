# Lean implementation rules

For every coding task:

1. Understand the existing execution path before editing.
2. Do not implement functionality that is not required.
3. Reuse existing code before creating new code.
4. Prefer the standard library and native platform features.
5. Prefer an already-installed dependency over adding another dependency.
6. Modify the smallest reasonable number of files.
7. Avoid speculative abstractions, wrappers, configuration layers, factories,
   helpers, compatibility layers, and generalized frameworks.
8. Do not create a new component, service, hook, class, schema, or utility
   unless the existing structure cannot cleanly support the change.
9. Remove obsolete code made unnecessary by the change.
10. Preserve required validation, security, accessibility, data-loss protection,
    error handling, logging, and backwards compatibility.
11. Run the smallest relevant tests first, followed by the required project
    checks.
12. Report the files changed, tests run, and any remaining verified limitation.

Prefer the simplest complete implementation, not the shortest unsafe implementation.
Do not rewrite unrelated code.
Do not introduce a dependency when the platform or existing codebase already
provides the required capability.