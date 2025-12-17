#!/usr/bin/env python3
"""
reposearch Web UI - Flask app for searching PRs with instrumented tests
"""

from flask import Flask, render_template, request, jsonify
import threading
import uuid
from reposearch import (
    search_repo, GitHubClient, analyze_pr, apply_filters,
    FilterConfig, load_repos, to_json
)
from dataclasses import asdict

app = Flask(__name__)

# Store search results and status
searches = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/repos")
def get_repos():
    """Get master repo list."""
    repos = load_repos()
    return jsonify([asdict(r) for r in repos])


@app.route("/api/search", methods=["POST"])
def start_search():
    data = request.json
    repo = data.get("repo", "")
    limit = int(data.get("limit", 100))
    token = data.get("token", "")

    # Get filter options
    filters = FilterConfig(
        min_lines=int(data.get("minLines", 0)),
        max_lines=int(data.get("maxLines", 999999)),
        min_files=int(data.get("minFiles", 0)),
        max_files=int(data.get("maxFiles", 999999)),
        min_test_lines=int(data.get("minTestLines", 0)),
        added_only=data.get("addedOnly", False),
        require_implementation=data.get("requireImpl", False),
        min_score=int(data.get("minScore", 0)),
    )

    if "/" not in repo:
        return jsonify({"error": "Invalid repo format. Use owner/repo"}), 400

    search_id = str(uuid.uuid4())[:8]
    searches[search_id] = {
        "status": "running",
        "progress": 0,
        "total": 0,
        "results": [],
        "filtered_results": [],
        "repo": repo,
        "filters": asdict(filters)
    }

    # Run search in background
    def run_search():
        owner, repo_name = repo.split("/", 1)
        client = GitHubClient(token if token else None)

        try:
            prs = client.get_merged_prs(owner, repo_name, limit)
            searches[search_id]["total"] = len(prs)

            results = []
            for i, pr in enumerate(prs):
                result = analyze_pr(client, owner, repo_name, pr)
                if result:
                    results.append(result)

                searches[search_id]["progress"] = i + 1

                # Apply filters and convert to JSON for real-time updates
                filtered = apply_filters(results, filters)
                searches[search_id]["results"] = to_json(results)
                searches[search_id]["filtered_results"] = to_json(filtered)

            searches[search_id]["status"] = "complete"

        except Exception as e:
            searches[search_id]["status"] = "error"
            searches[search_id]["error"] = str(e)

    thread = threading.Thread(target=run_search)
    thread.start()

    return jsonify({"search_id": search_id})


@app.route("/api/search/<search_id>")
def get_search_status(search_id):
    if search_id not in searches:
        return jsonify({"error": "Search not found"}), 404

    return jsonify(searches[search_id])


@app.route("/api/search/<search_id>/refilter", methods=["POST"])
def refilter_results(search_id):
    """Re-apply filters to existing results without re-fetching."""
    if search_id not in searches:
        return jsonify({"error": "Search not found"}), 404

    data = request.json
    filters = FilterConfig(
        min_lines=int(data.get("minLines", 0)),
        max_lines=int(data.get("maxLines", 999999)),
        min_files=int(data.get("minFiles", 0)),
        max_files=int(data.get("maxFiles", 999999)),
        min_test_lines=int(data.get("minTestLines", 0)),
        added_only=data.get("addedOnly", False),
        require_implementation=data.get("requireImpl", False),
        min_score=int(data.get("minScore", 0)),
    )

    # Get raw results and re-filter
    # Note: We need to reconstruct PRResult objects, but for simplicity
    # we'll do client-side filtering instead
    return jsonify({"filters": asdict(filters)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
