# Contributing to Watchdog

Thank you for considering contributing to Watchdog! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Respect differing viewpoints and experiences
- Accept responsibility for mistakes

### Unacceptable Behavior

- Harassment or discriminatory language
- Trolling or insulting comments
- Public or private harassment
- Publishing others' private information
- Other conduct inappropriate in a professional setting

## Getting Started

### Prerequisites

- Raspberry Pi 5 with Hailo-8/8L (for hardware testing)
- Python 3.9+
- Git knowledge
- Familiarity with computer vision concepts (helpful)

### Development Environment

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/watchdog.git
cd watchdog

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/watchdog.git

# Create development branch
git checkout -b feature/your-feature-name
```

## Development Setup

### Install Development Dependencies

```bash
# Install main dependencies
pip3 install -r requirements.txt

# Install development tools
pip3 install black flake8 pytest pylint

# Install pre-commit hooks (optional)
pip3 install pre-commit
pre-commit install
```

### Configure Test Environment

```bash
# Copy test configuration
cp .env.example .env.test

# Use test credentials (Telegram test bot, etc.)
nano .env.test
```

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

1. **Bug Reports**
   - Use GitHub Issues
   - Include system information (Pi model, OS version, etc.)
   - Provide steps to reproduce
   - Include error logs

2. **Feature Requests**
   - Explain the use case
   - Describe expected behavior
   - Consider implementation complexity

3. **Documentation**
   - Fix typos or clarify instructions
   - Add examples
   - Improve README or guides

4. **Code Contributions**
   - Bug fixes
   - New features
   - Performance improvements
   - Code refactoring

### Reporting Bugs

**Before submitting:**
- Check existing issues
- Verify it's reproducible
- Test on latest version

**Bug report should include:**
```markdown
## Environment
- Watchdog version:
- Raspberry Pi model:
- OS version:
- Python version:
- Hailo SDK version:

## Description
Brief description of the issue

## Steps to Reproduce
1. Step one
2. Step two
3. ...

## Expected Behavior
What should happen

## Actual Behavior
What actually happened

## Logs
```
Relevant error messages or logs
```

## Additional Context
Screenshots, videos, or other helpful information
```

### Suggesting Enhancements

**Enhancement proposal should include:**
- Clear use case
- Expected benefit
- Proposed implementation (if known)
- Potential drawbacks or considerations

## Coding Standards

### Python Style Guide

Follow PEP 8 with these specifics:

```python
# Line length: 100 characters max
# Indentation: 4 spaces
# Quotes: Single quotes for strings, double for docstrings

# Good example
def process_detection(frame, confidence_threshold=0.45):
    """
    Process video frame for person detection.

    Args:
        frame: Input video frame (numpy array)
        confidence_threshold: Minimum confidence score (float)

    Returns:
        list: Detected persons with bounding boxes
    """
    detections = []
    # Implementation...
    return detections
```

### Code Organization

```python
# File structure:
1. Shebang and encoding
2. Module docstring
3. Imports (standard library, third-party, local)
4. Constants
5. Classes
6. Functions
7. Main execution block

# Example:
#!/usr/bin/env python3
"""
Module for handling alert notifications.
"""

import os
import sys
from datetime import datetime

import requests
from twilio.rest import Client

ALERT_COOLDOWN = 25

class AlertManager:
    pass

def send_alert():
    pass

if __name__ == "__main__":
    main()
```

### Naming Conventions

```python
# Constants: UPPER_CASE
MAX_RETRY_ATTEMPTS = 3

# Classes: PascalCase
class VideoProcessor:
    pass

# Functions/methods: snake_case
def process_video_frame():
    pass

# Variables: snake_case
detection_threshold = 0.45
```

### Documentation

```python
# All public functions must have docstrings
def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union between two bounding boxes.

    Args:
        box1 (tuple): First box (x1, y1, x2, y2)
        box2 (tuple): Second box (x1, y1, x2, y2)

    Returns:
        float: IOU score between 0.0 and 1.0

    Raises:
        ValueError: If boxes are invalid
    """
    pass
```

## Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_detection.py

# Run with coverage
pytest --cov=. tests/
```

### Writing Tests

```python
# tests/test_detection.py
import pytest
from detection import process_detection

def test_detection_with_valid_frame():
    """Test detection on valid input frame."""
    frame = create_test_frame()
    detections = process_detection(frame)
    assert len(detections) >= 0

def test_detection_with_invalid_frame():
    """Test detection handles invalid input."""
    with pytest.raises(ValueError):
        process_detection(None)
```

### Test Coverage

- Aim for >80% code coverage
- Test edge cases
- Test error handling
- Include integration tests for critical paths

## Submitting Changes

### Pull Request Process

1. **Update Your Fork**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Make Your Changes**
   ```bash
   # Create feature branch
   git checkout -b feature/awesome-feature

   # Make changes
   # ...

   # Commit with descriptive message
   git commit -m "Add awesome feature for X"
   ```

3. **Test Your Changes**
   ```bash
   # Run tests
   pytest tests/

   # Check code style
   flake8 .
   black --check .

   # Test on actual hardware (if possible)
   python3 hailo_theft_prevention.py
   ```

4. **Push to Your Fork**
   ```bash
   git push origin feature/awesome-feature
   ```

5. **Create Pull Request**
   - Go to GitHub
   - Click "New Pull Request"
   - Select your branch
   - Fill out PR template

### Pull Request Guidelines

**PR Title Format:**
```
[Type] Brief description

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation changes
- style: Code style changes (formatting)
- refactor: Code refactoring
- test: Adding/updating tests
- chore: Maintenance tasks

Examples:
- feat: Add multi-camera support
- fix: Resolve RTSP reconnection issue
- docs: Update installation guide
```

**PR Description Template:**
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Tested on Raspberry Pi 5
- [ ] Tested with Hailo-8
- [ ] All tests pass
- [ ] No new warnings

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No breaking changes (or documented)

## Related Issues
Fixes #123
Related to #456
```

### Review Process

1. **Automated Checks**
   - CI/CD pipeline runs tests
   - Code style validation
   - Security scans

2. **Code Review**
   - Maintainer reviews code
   - Feedback provided
   - Requested changes made

3. **Approval and Merge**
   - PR approved by maintainer
   - Merged to main branch
   - Contributor acknowledged

### Commit Message Guidelines

```bash
# Good commit messages
git commit -m "Fix memory leak in video processing loop"
git commit -m "Add support for RTSP over HTTPS"
git commit -m "Update README with troubleshooting section"

# Bad commit messages
git commit -m "Fixed stuff"
git commit -m "WIP"
git commit -m "asdf"
```

## Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes
- GitHub contributors page

## Questions?

- Open a discussion on GitHub
- Check existing issues and PRs
- Review documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Watchdog!
