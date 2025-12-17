#!/usr/bin/env python3
"""
reposearch - Find PRs that add instrumented tests to Android repos

Usage:
    python reposearch.py <owner/repo> [options]
    python reposearch.py --list                    # List repos from master list
    python reposearch.py --from-list <name>        # Search a repo from master list

Examples:
    python reposearch.py android/nowinandroid
    python reposearch.py android/nowinandroid --limit 50
    python reposearch.py android/nowinandroid --min-lines 50 --min-files 3
    python reposearch.py --from-list nowinandroid --added-only
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import Optional
from pathlib import Path
import requests

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ============================================================================
# Cache System
# ============================================================================

CACHE_DIR = Path(__file__).parent / "cache"


def get_cache_path(owner: str, repo: str) -> Path:
    """Get cache file path for a repo."""
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{owner}_{repo}.json"


def load_cache(owner: str, repo: str) -> dict:
    """Load cached PR data for a repo."""
    cache_path = get_cache_path(owner, repo)
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {"prs": {}, "last_updated": None}


def save_cache(owner: str, repo: str, cache_data: dict):
    """Save PR data to cache."""
    cache_path = get_cache_path(owner, repo)
    cache_data["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(cache_path, "w") as f:
        json.dump(cache_data, f, indent=2)


def clear_cache(owner: str = None, repo: str = None):
    """Clear cache for a specific repo or all repos."""
    if owner and repo:
        cache_path = get_cache_path(owner, repo)
        if cache_path.exists():
            cache_path.unlink()
            print(f"Cleared cache for {owner}/{repo}")
    else:
        if CACHE_DIR.exists():
            for f in CACHE_DIR.glob("*.json"):
                f.unlink()
            print("Cleared all cache")


def list_cache():
    """List all cached repos."""
    if not CACHE_DIR.exists():
        print("No cache found")
        return

    print("\n" + "="*60)
    print("CACHED REPOSITORIES")
    print("="*60 + "\n")

    for f in sorted(CACHE_DIR.glob("*.json")):
        with open(f) as fp:
            data = json.load(fp)
        repo_name = f.stem.replace("_", "/")
        pr_count = len(data.get("prs", {}))
        updated = data.get("last_updated", "unknown")
        print(f"  {repo_name}: {pr_count} PRs cached (updated: {updated})")

    print()


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class FileChange:
    path: str
    additions: int
    deletions: int
    status: str  # added, modified, removed

    @property
    def total_changes(self) -> int:
        return self.additions + self.deletions


@dataclass
class PRResult:
    number: int
    title: str
    url: str
    author: str
    merged_at: str

    # All files in the PR
    all_files: list[FileChange] = field(default_factory=list)

    # Categorized files
    test_files_added: list[FileChange] = field(default_factory=list)
    test_files_modified: list[FileChange] = field(default_factory=list)
    src_files_added: list[FileChange] = field(default_factory=list)
    src_files_modified: list[FileChange] = field(default_factory=list)

    @property
    def total_lines_changed(self) -> int:
        return sum(f.total_changes for f in self.all_files)

    @property
    def total_additions(self) -> int:
        return sum(f.additions for f in self.all_files)

    @property
    def total_deletions(self) -> int:
        return sum(f.deletions for f in self.all_files)

    @property
    def total_files_changed(self) -> int:
        return len(self.all_files)

    @property
    def test_lines_added(self) -> int:
        return sum(f.additions for f in self.test_files_added)

    @property
    def has_implementation(self) -> bool:
        """PR has both source AND test changes (not test-only)."""
        has_tests = bool(self.test_files_added or self.test_files_modified)
        has_src = bool(self.src_files_added or self.src_files_modified)
        return has_tests and has_src

    @property
    def quality_score(self) -> int:
        """Score PR for task-worthiness (higher = better candidate)."""
        score = 0

        # Adds new test files (+10 each, max 30)
        score += min(len(self.test_files_added) * 10, 30)

        # Has both implementation and tests (+15)
        if self.has_implementation:
            score += 15

        # Multiple files changed (+5)
        if self.total_files_changed >= 3:
            score += 5

        # Sweet spot for lines changed (50-300 is ideal)
        if 50 <= self.total_lines_changed <= 300:
            score += 10
        elif 30 <= self.total_lines_changed <= 500:
            score += 5

        # Substantial test additions (+5)
        if self.test_lines_added >= 30:
            score += 5

        return score


@dataclass
class RepoConfig:
    name: str  # owner/repo
    license: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    stars: int = 0


@dataclass
class FilterConfig:
    min_lines: int = 0
    max_lines: int = 999999
    min_files: int = 0
    max_files: int = 999999
    min_test_lines: int = 0
    added_only: bool = False
    require_implementation: bool = False
    min_score: int = 0


# ============================================================================
# GitHub Client
# ============================================================================

class GitHubClient:
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        self.session = requests.Session()
        if self.token:
            self.session.headers["Authorization"] = f"token {self.token}"
        self.session.headers["Accept"] = "application/vnd.github.v3+json"
        self.session.headers["User-Agent"] = "reposearch"

    def _request(self, url: str, params: dict = None) -> dict | list:
        resp = self.session.get(url, params=params)

        # Handle rate limiting
        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            reset_time = int(resp.headers.get("X-RateLimit-Reset", 0))
            wait = max(reset_time - time.time(), 60)
            print(f"Rate limited. Waiting {wait:.0f}s...")
            time.sleep(wait)
            return self._request(url, params)

        resp.raise_for_status()
        return resp.json()

    def get_merged_prs(self, owner: str, repo: str, limit: int = 100) -> list[dict]:
        """Fetch merged PRs from a repo."""
        prs = []
        page = 1
        per_page = min(limit, 100)

        while len(prs) < limit:
            url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
            params = {
                "state": "closed",
                "sort": "updated",
                "direction": "desc",
                "per_page": per_page,
                "page": page
            }

            batch = self._request(url, params)
            if not batch:
                break

            # Filter to only merged PRs
            merged = [pr for pr in batch if pr.get("merged_at")]
            prs.extend(merged)

            print(f"Fetched page {page}, {len(merged)} merged PRs (total: {len(prs)})")

            if len(batch) < per_page:
                break
            page += 1

        return prs[:limit]

    def get_pr_files(self, owner: str, repo: str, pr_number: int) -> list[dict]:
        """Get files changed in a PR."""
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files"
        files = []
        page = 1

        while True:
            batch = self._request(url, {"per_page": 100, "page": page})
            if not batch:
                break
            files.extend(batch)
            if len(batch) < 100:
                break
            page += 1

        return files

    def get_repo_info(self, owner: str, repo: str) -> dict:
        """Get repository metadata."""
        url = f"https://api.github.com/repos/{owner}/{repo}"
        return self._request(url)


# ============================================================================
# File Classification
# ============================================================================

def is_instrumented_test_file(path: str) -> bool:
    """Check if a file path is an Android instrumented test."""
    if "/androidTest/" not in path and "\\androidTest\\" not in path:
        return False
    if not (path.endswith(".kt") or path.endswith(".java")):
        return False
    return True


def is_source_file(path: str) -> bool:
    """Check if a file is a main source file (not test, not config)."""
    # Must be Kotlin or Java
    if not (path.endswith(".kt") or path.endswith(".java")):
        return False

    # Exclude test directories
    test_dirs = ["/test/", "/androidTest/", "/testShared/", "/sharedTest/"]
    if any(d in path for d in test_dirs):
        return False

    # Should be in src/main or similar
    if "/src/" in path and "/main/" in path:
        return True

    # Or just in a source directory
    if "/src/" in path:
        return True

    return False


def classify_file(f: dict) -> tuple[FileChange, str]:
    """Classify a file and return (FileChange, category)."""
    path = f["filename"]
    file_change = FileChange(
        path=path,
        additions=f.get("additions", 0),
        deletions=f.get("deletions", 0),
        status=f.get("status", "unknown")
    )

    if is_instrumented_test_file(path):
        return file_change, "test"
    elif is_source_file(path):
        return file_change, "src"
    else:
        return file_change, "other"


# ============================================================================
# PR Analysis
# ============================================================================

def analyze_pr(client: GitHubClient, owner: str, repo: str, pr: dict) -> Optional[PRResult]:
    """Analyze a PR and categorize all its files."""
    pr_number = pr["number"]
    files = client.get_pr_files(owner, repo, pr_number)

    result = PRResult(
        number=pr_number,
        title=pr["title"],
        url=pr["html_url"],
        author=pr["user"]["login"],
        merged_at=pr["merged_at"],
    )

    for f in files:
        file_change, category = classify_file(f)
        result.all_files.append(file_change)

        if category == "test":
            if file_change.status == "added":
                result.test_files_added.append(file_change)
            elif file_change.status in ("modified", "renamed"):
                result.test_files_modified.append(file_change)
        elif category == "src":
            if file_change.status == "added":
                result.src_files_added.append(file_change)
            elif file_change.status in ("modified", "renamed"):
                result.src_files_modified.append(file_change)

    # Only return PRs that touch test files
    if not result.test_files_added and not result.test_files_modified:
        return None

    return result


def apply_filters(results: list[PRResult], filters: FilterConfig) -> list[PRResult]:
    """Apply all filters to results."""
    filtered = []

    for r in results:
        # Lines changed filter
        if r.total_lines_changed < filters.min_lines:
            continue
        if r.total_lines_changed > filters.max_lines:
            continue

        # Files changed filter
        if r.total_files_changed < filters.min_files:
            continue
        if r.total_files_changed > filters.max_files:
            continue

        # Test lines filter
        if r.test_lines_added < filters.min_test_lines:
            continue

        # Added only filter
        if filters.added_only and not r.test_files_added:
            continue

        # Require implementation (src + test)
        if filters.require_implementation and not r.has_implementation:
            continue

        # Minimum quality score
        if r.quality_score < filters.min_score:
            continue

        filtered.append(r)

    return filtered


# ============================================================================
# Master Repo List
# ============================================================================

DEFAULT_REPOS = [
    RepoConfig(
        name="android/nowinandroid",
        license="Apache-2.0",
        description="Fully functional Android app built with Kotlin and Jetpack Compose",
        tags=["compose", "hilt", "room", "coroutines", "flow"]
    ),
    RepoConfig(
        name="android/architecture-samples",
        license="Apache-2.0",
        description="To-Do app demonstrating Android architecture best practices",
        tags=["compose", "hilt", "mvvm", "room"]
    ),
    RepoConfig(
        name="android/sunflower",
        license="Apache-2.0",
        description="Gardening app illustrating Android development best practices",
        tags=["compose", "hilt", "room", "navigation"]
    ),
    RepoConfig(
        name="thunderbird/thunderbird-android",
        license="Apache-2.0",
        description="Thunderbird/K-9 Mail email client for Android",
        tags=["email", "large", "production", "kotlin"]
    ),
    RepoConfig(
        name="home-assistant/android",
        license="Apache-2.0",
        description="Home Assistant Android companion app",
        tags=["compose", "hilt", "production", "iot"]
    ),
    RepoConfig(
        name="android/compose-samples",
        license="Apache-2.0",
        description="Official Jetpack Compose samples",
        tags=["compose", "samples", "material3"]
    ),
    RepoConfig(
        name="JetBrains/compose-multiplatform",
        license="Apache-2.0",
        description="Compose Multiplatform framework",
        tags=["compose", "kmp", "multiplatform"]
    ),
    RepoConfig(
        name="mikepenz/AboutLibraries",
        license="Apache-2.0",
        description="Automatically collect dependencies and licenses",
        tags=["compose", "library", "licenses"]
    ),
]


def get_repos_dir() -> Path:
    """Get the directory containing repo configs."""
    return Path(__file__).parent / "data"


def load_repos() -> list[RepoConfig]:
    """Load repos from YAML file or return defaults."""
    repos_file = get_repos_dir() / "repos.yaml"

    if repos_file.exists() and HAS_YAML:
        with open(repos_file) as f:
            data = yaml.safe_load(f)
            return [RepoConfig(**r) for r in data.get("repos", [])]

    return DEFAULT_REPOS


def save_repos(repos: list[RepoConfig]):
    """Save repos to YAML file."""
    if not HAS_YAML:
        print("Warning: PyYAML not installed, cannot save repos")
        return

    repos_dir = get_repos_dir()
    repos_dir.mkdir(exist_ok=True)

    repos_file = repos_dir / "repos.yaml"
    with open(repos_file, "w") as f:
        yaml.dump({"repos": [asdict(r) for r in repos]}, f, default_flow_style=False)


def find_repo_by_name(name: str) -> Optional[RepoConfig]:
    """Find a repo by partial name match."""
    repos = load_repos()
    name_lower = name.lower()

    for repo in repos:
        # Exact match on full name
        if repo.name.lower() == name_lower:
            return repo
        # Match on repo name only (without owner)
        repo_name = repo.name.split("/")[-1].lower()
        if repo_name == name_lower:
            return repo
        # Partial match
        if name_lower in repo.name.lower():
            return repo

    return None


def list_repos():
    """Print all repos in the master list."""
    repos = load_repos()

    print("\n" + "="*80)
    print("MASTER REPO LIST")
    print("="*80 + "\n")

    for r in repos:
        tags = ", ".join(r.tags) if r.tags else "none"
        print(f"  {r.name}")
        print(f"    License: {r.license}")
        print(f"    Tags: {tags}")
        if r.description:
            print(f"    {r.description}")
        print()


# ============================================================================
# Search & Output
# ============================================================================

def pr_result_from_dict(d: dict) -> PRResult:
    """Reconstruct PRResult from cached dict."""
    return PRResult(
        number=d["number"],
        title=d["title"],
        url=d["url"],
        author=d["author"],
        merged_at=d["merged_at"],
        all_files=[FileChange(**f) for f in d.get("all_files", [])],
        test_files_added=[FileChange(**f) for f in d.get("test_files_added", [])],
        test_files_modified=[FileChange(**f) for f in d.get("test_files_modified", [])],
        src_files_added=[FileChange(**f) for f in d.get("src_files_added", [])],
        src_files_modified=[FileChange(**f) for f in d.get("src_files_modified", [])],
    )


def search_repo(owner: str, repo: str, limit: int = 100, token: str = None,
                use_cache: bool = True, refresh_cache: bool = False) -> list[PRResult]:
    """Search a repo for PRs that add instrumented tests."""
    client = GitHubClient(token)

    # Load existing cache
    cache = load_cache(owner, repo) if use_cache else {"prs": {}}
    cached_prs = cache.get("prs", {})

    if cached_prs and not refresh_cache:
        print(f"Found {len(cached_prs)} cached PRs for {owner}/{repo}")

    print(f"Fetching merged PRs from {owner}/{repo}...")
    prs = client.get_merged_prs(owner, repo, limit)
    print(f"Found {len(prs)} merged PRs, analyzing...\n")

    results = []
    new_cached = 0
    from_cache = 0

    for i, pr in enumerate(prs):
        pr_key = str(pr["number"])

        # Check if we have this PR cached
        if pr_key in cached_prs and use_cache and not refresh_cache:
            cached_data = cached_prs[pr_key]
            if cached_data:  # PR has test files
                result = pr_result_from_dict(cached_data)
                results.append(result)
                from_cache += 1
                print(f"[{i+1}/{len(prs)}] PR #{pr['number']}: (cached) score={result.quality_score}")
            else:
                print(f"[{i+1}/{len(prs)}] PR #{pr['number']}: (cached) no instrumented tests")
                from_cache += 1
            continue

        # Analyze PR (not cached)
        result = analyze_pr(client, owner, repo, pr)

        if result:
            results.append(result)
            # Cache the result
            cached_prs[pr_key] = asdict(result)
            new_cached += 1
            added = len(result.test_files_added)
            modified = len(result.test_files_modified)
            score = result.quality_score
            print(f"[{i+1}/{len(prs)}] PR #{result.number}: +{added} added, ~{modified} modified, score={score}")
        else:
            # Cache that this PR has no tests (so we don't re-check)
            cached_prs[pr_key] = None
            new_cached += 1
            print(f"[{i+1}/{len(prs)}] PR #{pr['number']}: no instrumented tests")

    # Save updated cache
    if use_cache and new_cached > 0:
        cache["prs"] = cached_prs
        save_cache(owner, repo, cache)
        print(f"\nCache updated: {new_cached} new, {from_cache} from cache")

    return results


def print_results(results: list[PRResult], sort_by: str = "score"):
    """Print results in a readable format."""
    print("\n" + "="*80)
    print(f"FOUND {len(results)} PRs WITH INSTRUMENTED TESTS")
    print("="*80 + "\n")

    # Sort results
    if sort_by == "score":
        results.sort(key=lambda r: r.quality_score, reverse=True)
    elif sort_by == "lines":
        results.sort(key=lambda r: r.total_lines_changed, reverse=True)
    elif sort_by == "tests":
        results.sort(key=lambda r: len(r.test_files_added), reverse=True)

    for r in results:
        impl_badge = "[IMPL+TEST]" if r.has_implementation else "[TEST-ONLY]"
        print(f"PR #{r.number}: {r.title}")
        print(f"   URL: {r.url}")
        print(f"   Author: {r.author} | Merged: {r.merged_at}")
        print(f"   Stats: {r.total_files_changed} files, +{r.total_additions}/-{r.total_deletions} lines")
        print(f"   Score: {r.quality_score} {impl_badge}")

        if r.test_files_added:
            print(f"   Tests ADDED ({len(r.test_files_added)}):")
            for t in r.test_files_added:
                print(f"      + {t.path} (+{t.additions}/-{t.deletions})")

        if r.test_files_modified:
            print(f"   Tests MODIFIED ({len(r.test_files_modified)}):")
            for t in r.test_files_modified:
                print(f"      ~ {t.path} (+{t.additions}/-{t.deletions})")

        if r.src_files_added:
            print(f"   Source ADDED ({len(r.src_files_added)}):")
            for t in r.src_files_added[:5]:  # Limit to first 5
                print(f"      + {t.path} (+{t.additions}/-{t.deletions})")
            if len(r.src_files_added) > 5:
                print(f"      ... and {len(r.src_files_added) - 5} more")

        print()


def to_json(results: list[PRResult]) -> list[dict]:
    """Convert results to JSON-serializable format with computed properties."""
    output = []
    for r in results:
        d = asdict(r)
        # Add computed properties
        d["total_lines_changed"] = r.total_lines_changed
        d["total_files_changed"] = r.total_files_changed
        d["test_lines_added"] = r.test_lines_added
        d["has_implementation"] = r.has_implementation
        d["quality_score"] = r.quality_score
        output.append(d)
    return output


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Find PRs that add instrumented tests to Android repos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s android/nowinandroid
  %(prog)s android/nowinandroid --min-lines 50 --min-files 3
  %(prog)s android/nowinandroid --require-impl --min-score 20
  %(prog)s --list
  %(prog)s --from-list nowinandroid --added-only
        """
    )

    # Repo selection
    parser.add_argument(
        "repo",
        nargs="?",
        help="Repository in owner/repo format (e.g., android/nowinandroid)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all repos in master list"
    )
    parser.add_argument(
        "--from-list", "-f",
        metavar="NAME",
        help="Search a repo from the master list by name"
    )

    # Cache options
    cache_group = parser.add_argument_group("cache")
    cache_group.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't use cache, fetch everything fresh"
    )
    cache_group.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh cache (re-fetch all PRs)"
    )
    cache_group.add_argument(
        "--list-cache",
        action="store_true",
        help="List all cached repos"
    )
    cache_group.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cache for the specified repo (or all if no repo)"
    )

    # Limits
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=100,
        help="Max number of PRs to analyze (default: 100)"
    )

    # Filters
    filter_group = parser.add_argument_group("filters")
    filter_group.add_argument(
        "--min-lines",
        type=int,
        default=0,
        help="Minimum total lines changed"
    )
    filter_group.add_argument(
        "--max-lines",
        type=int,
        default=999999,
        help="Maximum total lines changed"
    )
    filter_group.add_argument(
        "--min-files",
        type=int,
        default=0,
        help="Minimum files changed"
    )
    filter_group.add_argument(
        "--max-files",
        type=int,
        default=999999,
        help="Maximum files changed"
    )
    filter_group.add_argument(
        "--min-test-lines",
        type=int,
        default=0,
        help="Minimum lines added in test files"
    )
    filter_group.add_argument(
        "--added-only", "-a",
        action="store_true",
        help="Only PRs that ADD new test files (not just modify)"
    )
    filter_group.add_argument(
        "--require-impl", "-i",
        action="store_true",
        help="Only PRs with both implementation AND test changes"
    )
    filter_group.add_argument(
        "--min-score", "-s",
        type=int,
        default=0,
        help="Minimum quality score"
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        help="Output results to JSON file"
    )
    parser.add_argument(
        "--sort",
        choices=["score", "lines", "tests"],
        default="score",
        help="Sort results by (default: score)"
    )
    parser.add_argument(
        "--token", "-t",
        help="GitHub token (or set GITHUB_TOKEN env var)"
    )

    args = parser.parse_args()

    # Handle --list
    if args.list:
        list_repos()
        return

    # Handle --list-cache
    if args.list_cache:
        list_cache()
        return

    # Handle --clear-cache
    if args.clear_cache:
        if args.repo:
            owner, repo = args.repo.split("/", 1)
            clear_cache(owner, repo)
        else:
            clear_cache()
        return

    # Determine repo to search
    repo_str = None
    if args.from_list:
        repo_config = find_repo_by_name(args.from_list)
        if not repo_config:
            print(f"Error: repo '{args.from_list}' not found in master list")
            print("Use --list to see available repos")
            sys.exit(1)
        repo_str = repo_config.name
        print(f"Using repo from master list: {repo_str}")
    elif args.repo:
        repo_str = args.repo
    else:
        parser.print_help()
        sys.exit(1)

    # Parse owner/repo
    if "/" not in repo_str:
        print("Error: repo must be in owner/repo format")
        sys.exit(1)

    owner, repo = repo_str.split("/", 1)

    # Build filter config
    filters = FilterConfig(
        min_lines=args.min_lines,
        max_lines=args.max_lines,
        min_files=args.min_files,
        max_files=args.max_files,
        min_test_lines=args.min_test_lines,
        added_only=args.added_only,
        require_implementation=args.require_impl,
        min_score=args.min_score,
    )

    # Run search
    results = search_repo(
        owner, repo, args.limit, args.token,
        use_cache=not args.no_cache,
        refresh_cache=args.refresh
    )

    # Apply filters
    original_count = len(results)
    results = apply_filters(results, filters)

    if original_count != len(results):
        print(f"\nFiltered: {original_count} -> {len(results)} PRs")

    # Output
    print_results(results, sort_by=args.sort)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(to_json(results), f, indent=2, default=str)
        print(f"Results saved to {args.output}")

    # Summary
    total_added = sum(len(r.test_files_added) for r in results)
    total_modified = sum(len(r.test_files_modified) for r in results)
    impl_count = sum(1 for r in results if r.has_implementation)
    print(f"\nSummary: {len(results)} PRs, {total_added} tests added, {total_modified} modified")
    print(f"         {impl_count} PRs have both implementation + tests")


if __name__ == "__main__":
    main()
