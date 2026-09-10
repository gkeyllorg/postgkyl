---
name: postgkyl-testing
description: Design and best practices for testing Postgkyl code, including unit tests and examples.
---

# Testing

Unit tests aim for 100% code coverage where possible. Every change must have an associated unit test. Every bug fix must have a test. Every feature must be tested. Run `pytest` to check that all tests pass.

Data for testing must be generated automatically by `tests/generate_test_data.py`. This keeps the repository light. DO NOT COMMIT LARGE DATA FILES TO GIT.

Examples are real-world use cases of postgkyl. They read data and make plots. Examples are displayed in the documentation.
