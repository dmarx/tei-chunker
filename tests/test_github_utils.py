# tests/test_github_utils.py
"""
Tests for GitHub integration utilities.
"""
import os
import pytest
from unittest.mock import Mock, patch
from github import Github
from scripts.github_utils import (
    get_github_context,
    get_feature_requests,
    handle_missing_features
)

@pytest.fixture
def mock_github_env(monkeypatch):
    """Set up mock GitHub environment variables."""
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")

@pytest.fixture
def mock_issue():
    """Create a mock GitHub issue."""
    issue = Mock()
    issue.number = 1
    issue.body = '''{
        "name": "test-feature",
        "inputs": {"input/test": "var"},
        "prompt": "Test prompt with {var}"
    }'''
    issue.state = "open"
    return issue

@pytest.fixture
def mock_repo(mock_issue):
    """Create a mock GitHub repository."""
    repo = Mock()
    repo.get_issues.return_value = [mock_issue]
    return repo

def test_github_context(mock_github_env):
    """Test extraction of GitHub context from environment."""
    owner, repo, token = get_github_context()
    assert owner == "owner"
    assert repo == "repo"
    assert token == "test-token"

def test_missing_github_env():
    """Test handling of missing GitHub environment variables."""
    with pytest.raises(RuntimeError):
        get_github_context()

@patch('github.Github')
def test_get_feature_requests(mock_github_class, mock_repo):
    """Test fetching feature requests from GitHub issues."""
    # Set up mock
    mock_github = Mock()
    mock_github.get_repo.return_value = mock_repo
    mock_github_class.return_value = mock_github
    
    # Get feature requests
    requests = list(get_feature_requests("owner", "repo"))
    assert len(requests) == 1
    
    request, issue = requests[0]
    assert request.name == "test-feature"
    assert "var" in request.inputs.values()
    assert issue.number == 1

def test_get_feature_requests_with_name(mock_repo):
    """Test fetching feature requests filtered by name."""
    with patch('github.Github') as mock_github_class:
        mock_github = Mock()
        mock_github.get_repo.return_value = mock_repo
        mock_github_class.return_value = mock_github
        
        # Get feature requests with name filter
        requests = list(get_feature_requests(
            "owner", "repo", feature_name="test-feature"
        ))
        
        # Verify correct labels were used
        mock_repo.get_issues.assert_called_with(
            labels=["feature-node", "feature:test-feature"],
            state="all"
        )

def test_invalid_feature_request(mock_repo):
    """Test handling of invalid feature request format."""
    # Modify mock issue to have invalid body
    mock_repo.get_issues.return_value[0].body = "invalid json"
    
    with patch('github.Github') as mock_github_class:
        mock_github = Mock()
        mock_github.get_repo.return_value = mock_repo
        mock_github_class.return_value = mock_github
        
        # Get feature requests
        requests = list(get_feature_requests("owner", "repo"))
        assert len(requests) == 0  # Invalid request should be skipped

def test_handle_missing_features(mock_repo):
    """Test reopening of issues for missing features."""
    with patch('github.Github') as mock_github_class:
        mock_github = Mock()
        mock_github.get_repo.return_value = mock_repo
        mock_github_class.return_value = mock_github
        
        # Handle missing features
        missing = {"test-feature"}
        handle_missing_features("owner", "repo", missing)
        
        # Verify issue was reopened
        mock_repo.get_issues.return_value[0].edit.assert_called_with(
            state="open"
        )

def test_handle_nonexistent_feature(mock_repo):
    """Test handling of missing features with no corresponding issues."""
    with patch('github.Github') as mock_github_class:
        mock_github = Mock()
        mock_github.get_repo.return_value = mock_repo
        mock_github_class.return_value = mock_github
        
        # Handle nonexistent feature
        missing = {"nonexistent-feature"}
        handle_missing_features("owner", "repo", missing)
        
        # Verify no issues were reopened
        assert not mock_repo.get_issues.return_value[0].edit.called

def test_github_api_error(mock_repo):
    """Test handling of GitHub API errors."""
    with patch('github.Github') as mock_github_class:
        mock_github = Mock()
        mock_github.get_repo.side_effect = Exception("API Error")
        mock_github_class.return_value = mock_github
        
        # Attempt to get feature requests
        requests = list(get_feature_requests("owner", "repo"))
        assert len(requests) == 0  # Should handle error gracefully
