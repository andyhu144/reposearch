# RepoSearch

Find PRs that add instrumented tests to Android repositories. Built to help discover task candidates for AI coding agents.

## Features

- Search any GitHub repo for PRs with instrumented tests
- Filter by lines changed, files changed, test coverage
- Quality scoring to find ideal task candidates
- Master repo list with curated Android projects
- Web UI and CLI interfaces
- Export results to JSON

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### CLI

```bash
# Search a repo
python reposearch.py android/nowinandroid --limit 100

# With filters
python reposearch.py android/nowinandroid --min-lines 50 --min-files 3 --require-impl

# List curated repos
python reposearch.py --list

# Search from curated list
python reposearch.py --from-list nowinandroid --added-only

# Export to JSON
python reposearch.py android/nowinandroid -o results.json
```

### Web UI

```bash
python app.py
# Open http://localhost:5000
```

## Filters

| Filter | Description |
|--------|-------------|
| `--min-lines` | Minimum total lines changed |
| `--max-lines` | Maximum total lines changed |
| `--min-files` | Minimum files changed |
| `--max-files` | Maximum files changed |
| `--min-test-lines` | Minimum lines added in test files |
| `--min-score` | Minimum quality score |
| `--added-only` | Only PRs that add new test files |
| `--require-impl` | Only PRs with both impl + test changes |

## Quality Score

PRs are scored based on:
- New test files added (+10 each, max 30)
- Has both implementation and tests (+15)
- Multiple files changed (+5)
- Lines in sweet spot 50-300 (+10)
- Substantial test additions (+5)

## License

MIT
