# Contributing

Thanks for your interest in improving this project!

## How to contribute

1. Fork the repo and create your branch from `main`:

   ```bash
   git checkout -b feature/my-feature
   ```

2. Install development dependencies:

   ```bash
   pip install -r requirements-dev.txt
   pre-commit install
   ```

3. Make changes following the existing code style. Before committing run:

   ```bash
   ruff check src tests app
   pytest
   ```

4. Open a Pull Request using the provided template.

## Commit style

Use concise, imperative commit messages:
`feat: add SMOTE handling`, `fix: guard division by zero in utilization`,
`docs: update README metrics table`.
