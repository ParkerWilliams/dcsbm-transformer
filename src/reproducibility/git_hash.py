"""Git hash capture with dirty-tree detection for code provenance tracking.

Stores the short git SHA with results so any experiment can be traced back
to the exact code version that produced it.
"""

import subprocess


def get_git_hash() -> str:
    """Get the short git SHA of the current commit with dirty detection.

    Returns the 7+ character short SHA of HEAD. If there are uncommitted
    changes (staged or unstaged), appends '-dirty' to flag unreproducible state.

    Returns:
        Git hash string in one of three forms:
        - "a3f9c1d" — clean working tree
        - "a3f9c1d-dirty" — uncommitted changes present
        - "unknown" — not in a git repository or git not available
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"

    # Check for unstaged changes
    dirty = False
    try:
        subprocess.check_output(
            ["git", "diff", "--quiet"],
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        dirty = True

    # Check for staged changes
    if not dirty:
        try:
            subprocess.check_output(
                ["git", "diff", "--quiet", "--cached"],
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            dirty = True

    if dirty:
        sha += "-dirty"

    return sha
