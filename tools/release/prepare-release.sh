#!/bin/bash

readonly VERSION_REGEX='__version__ = "([0-9.]+)"' # Regex to find the version string
readonly CHANGELOG_FILE="CHANGELOG.md"
readonly INIT_FILE="src/kagglehub/__init__.py"
readonly FEATURE_BRANCH_PREFIX="release-"
# Determine the root repo directory based on the location of this script
readonly ROOT_REPO_DIR=$(dirname $(dirname $(dirname $(realpath $0))))

# New version specified by the person preparing the release
NEW_VERSION=$1

# Exit if any command fails
set -e

# Run all the commands from repo root
cd "$ROOT_REPO_DIR"

# 1. Pull latest main
git checkout main
git pull origin main

# 2. Fetch latest tag
git fetch --tags

# 3. Determine commit hash of latest tag
latest_tag=$(git describe --tags --abbrev=0) # Gets the latest tag name
tag_hash=$(git rev-list -n 1 "$latest_tag")

# 4. Gather commits since that hash
commits=$(git log --pretty=format:"%s" "$tag_hash..main")

# 5. Update CHANGELOG.md
new_changelog_entry="\n"
if [[ -n "$commits" ]]; then
    date_string=$(date +"%B %d, %Y")
    new_changelog_entry+="## v$NEW_VERSION ($date_string)\n\n"

    while IFS= read -r line; do
        if [[ -n "$line" ]]; then
            pr_number=$(echo "$line" | grep -o '#[0-9]*' | sed 's/#//')
            pattern="\(#[0-9])"
            commit_message=$(echo "$line" | sed -E 's/ \(#[0-9]*\) *//')
            new_changelog_entry+="* ${commit_message} ([#${pr_number}](https://github.com/Kaggle/kagglehub/pull/${pr_number}))\n"
        fi
    done <<< "$commits"
else
  echo "No significant changes since last release"
  exit 1
fi

# Read the file, separate the first line (# Changelog)
first_line=$(head -n 1 "$CHANGELOG_FILE")
rest_of_file=$(tail -n +2 "$CHANGELOG_FILE") # Everything after the first line

# Combine: # Changelog + New Entry + Rest of the file
echo "$first_line" > "$CHANGELOG_FILE".tmp  # Write the first line
printf "$new_changelog_entry" >> "$CHANGELOG_FILE".tmp # Append the new entry
echo "$rest_of_file" >> "$CHANGELOG_FILE".tmp # Append the rest of the file

mv "$CHANGELOG_FILE".tmp "$CHANGELOG_FILE"

# 6. Update __init__.py version
sed -i -E "s/$VERSION_REGEX/__version__ = \"$NEW_VERSION\"/" "$INIT_FILE"

# 7. Create and push feature branch
feature_branch="$FEATURE_BRANCH_PREFIX$NEW_VERSION"
git checkout -b "$feature_branch"
git add "$CHANGELOG_FILE" "$INIT_FILE"
git commit -m "Prepare release $NEW_VERSION"
git push origin "$feature_branch"

echo "Release preparation complete. Go to https://github.com/Kaggle/kagglehub/compare/$feature_branch?expand=1 to open the PR."
