---
# Contributing to Intelligent RL-Based Routing System

First off, **thank you** for considering contributing to this project! 🎉

This document provides guidelines for contributing to the Intelligent RL-Based Routing System. Following these guidelines helps maintain code quality and makes it easier for maintainers to review and merge your contributions.
---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Your First Code Contribution](#your-first-code-contribution)
  - [Pull Requests](#pull-requests)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Commit Guidelines](#commit-guidelines)
- [Documentation](#documentation)
- [Community](#community)

---

## 📜 Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [sdutt081@gmail.com].

---

## 🤝 How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the [existing issues](https://github.com/shivanshpandeygp/intelligent_routing_system/issues) to avoid duplicates.

#### How to Submit a Good Bug Report

Create an issue and include as many details as possible:

**Bug Report Template:**

**Bug Description<br>**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:

1. Go to '...'
2. Click on '...'
3. Scroll down to '...'
4. See error

**Expected Behavior**
A clear description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment:**

- OS: [e.g., Windows 10, Ubuntu 22.04, macOS 14]
- Python Version: [e.g., 3.10.5]
- PyTorch Version: [e.g., 2.0.1]
- Streamlit Version: [e.g., 1.28.0]

**Additional Context**
Add any other context about the problem here.

**Logs**

Paste relevant error logs here

```
undefined
```

#### Example Bug Report

**Bug Description**
DQN training crashes with "CUDA out of memory" error on large networks.

**To Reproduce**

1. Create mesh network with 50 nodes
2. Start DQN training with 5000 episodes
3. Error occurs after ~500 episodes

**Expected Behavior**
Training should complete without memory errors.

**Environment:**

- OS: Windows 10
- Python: 3.10.5
- PyTorch: 2.0.1 (CUDA 11.8)
- GPU: NVIDIA RTX 3060 (12GB)

**Logs**

RuntimeError: CUDA out of memory. Tried to allocate 2.50 GiB

```

```

---

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

**Enhancement Template:**

```markdown
**Is your feature request related to a problem?**
A clear description of the problem. Ex. I'm always frustrated when [...]

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Any alternative solutions or features you've considered.

**Additional context**
Any other context or screenshots about the feature request.

**Proposed Implementation** (Optional)
If you have ideas on how to implement this, share them here.
```

#### Example Enhancement Request

```markdown
**Feature Request: Add A3C Algorithm**

**Is your feature request related to a problem?**
Current RL algorithms (Q-Learning, DQN) don't support parallel training on multi-core CPUs.

**Describe the solution you'd like**
Implement Asynchronous Advantage Actor-Critic (A3C) algorithm for:

- Faster training on multi-core systems
- Better exploration through parallel agents
- Improved convergence

**Proposed Implementation**

1. Create `a3c_routing.py` in backend/
2. Add threading support
3. Update UI to show parallel agent progress
4. Add documentation and tests

**References**

- Mnih et al. (2016) - Asynchronous Methods for Deep RL
```

---

### Your First Code Contribution

Unsure where to begin? Look for issues labeled:

- `good first issue` - Simple issues perfect for beginners
- `help wanted` - Issues where we need community help
- `documentation` - Improvements to docs
- `bug` - Confirmed bugs needing fixes

**Steps to contribute:**

1. **Comment on the issue** you want to work on
2. **Wait for assignment** from maintainers
3. **Fork the repository**
4. **Create a branch** for your changes
5. **Make your changes**
6. **Submit a pull request**

---

### Pull Requests

Follow these steps for a smooth PR process:

1. **Fork the repo** and create your branch from `main`
2. **Follow coding standards** (see below)
3. **Add tests** if you've added code
4. **Update documentation** if needed
5. **Ensure tests pass** locally
6. **Submit the PR** with a clear description

**PR Checklist:**

```markdown
## Pull Request Checklist

- [ ] Code follows project style guidelines
- [ ] Self-review of code completed
- [ ] Comments added to complex sections
- [ ] Documentation updated (if applicable)
- [ ] Tests added/updated (if applicable)
- [ ] All tests pass locally
- [ ] No merge conflicts with main branch
- [ ] Commit messages follow guidelines
- [ ] Screenshots added (for UI changes)
```

---

## 🛠️ Development Setup

### Prerequisites

- Python 3.8+
- Git
- Virtual environment tool

### Setup Steps

```bash
# 1. Fork and clone the repository
git clone https://github.com/shivanshpandeygp/intelligent_routing_system.git
cd intelligent_routing_system

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install development dependencies
pip install pytest pytest-cov black flake8 mypy

# 6. Run tests to verify setup
pytest tests/

# 7. Run the application
streamlit run frontend/app.py
```

### Project Structure

```
intelligent_routing_system/
├── backend/              # Core algorithms
│   ├── config.py         # Configuration
│   ├── q_learning_routing.py
│   ├── dqn_routing.py
│   └── ...
├── frontend/             # UI components
│   ├── app.py           # Main Streamlit app
│   └── ...
├── tests/               # Test files
│   ├── test_q_learning.py
│   └── ...
└── docs/                # Documentation
```

---

## 📝 Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with these modifications:

- **Line length**: 100 characters (instead of 79)
- **Indentation**: 4 spaces
- **Quotes**: Double quotes `"` for strings
- **Naming**:
  - Variables: `snake_case`
  - Constants: `UPPER_CASE`
  - Classes: `PascalCase`
  - Functions: `snake_case`

### Code Formatting

Use **Black** for automatic formatting:

```bash
# Format all files
black backend/ frontend/

# Check formatting without changes
black --check backend/ frontend/

# Format specific file
black backend/dqn_routing.py
```

### Linting

Use **Flake8** for style checking:

```bash
# Check all files
flake8 backend/ frontend/ --max-line-length=100

# Check specific file
flake8 backend/dqn_routing.py
```

Ignore these errors (add to `.flake8`):

```ini
[flake8]
max-line-length = 100
ignore = E203, W503
exclude = .venv, __pycache__
```

### Type Hints

Add type hints to all function signatures:

**Good:**

```python
def find_path(self, source: int, destination: int) -> Tuple[List[int], float, float]:
    """Find shortest path using trained agent."""
    pass
```

**Bad:**

```python
def find_path(self, source, destination):
    """Find shortest path."""
    pass
```

### Documentation Strings

Use **Google-style docstrings**:

```python
def train(self, source: int, destination: int, num_episodes: int = None) -> None:
    """
    Train the Q-Learning agent on specified route.

    Trains the agent to find optimal path from source to destination
    using Q-learning algorithm with epsilon-greedy exploration.

    Args:
        source: Source node ID (0 to num_nodes-1)
        destination: Destination node ID (0 to num_nodes-1)
        num_episodes: Number of training episodes (default: from config)

    Returns:
        None. Updates internal Q-table.

    Raises:
        ValueError: If source or destination not in network
        RuntimeError: If training fails to converge

    Example:
        >>> agent = QLearningRouting(network)
        >>> agent.train(source=0, destination=9, num_episodes=1000)
        >>> path, cost, time = agent.find_path(0, 9)
    """
    pass
```

### Code Comments

- **Explain WHY, not WHAT**
- Comment complex algorithms
- Add TODO comments for future work

**Good:**

```python
# Use Double DQN to reduce Q-value overestimation
# Standard DQN overestimates due to max operation
policy_q = self.policy_net(state)
target_q = self.target_net(state)
```

**Bad:**

```python
# Get Q values
policy_q = self.policy_net(state)
target_q = self.target_net(state)
```

---

## 🧪 Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=backend --cov=frontend

# Run specific test file
pytest tests/test_q_learning.py

# Run specific test
pytest tests/test_q_learning.py::test_training

# Run with verbose output
pytest -v

# Run and show print statements
pytest -s
```

### Writing Tests

Place tests in `tests/` directory with `test_` prefix:

```python
import pytest
from backend.q_learning_routing import QLearningRouting
from backend.graph_manager import NetworkGraph
from backend.config import Config

def test_q_learning_initialization():
    """Test Q-Learning agent initializes correctly."""
    network = NetworkGraph.create_mesh(num_nodes=5)
    config = Config()
    agent = QLearningRouting(network, config)

    assert agent.qtable is not None
    assert agent.epsilon == config.QL_EPSILON
    assert agent.total_training_episodes == 0

def test_q_learning_training():
    """Test Q-Learning training updates Q-table."""
    network = NetworkGraph.create_mesh(num_nodes=5)
    agent = QLearningRouting(network)

    # Train
    agent.train(source=0, destination=4, num_episodes=100)

    # Verify
    assert agent.total_training_episodes == 100
    assert len(agent.qtable) > 0

def test_q_learning_find_path():
    """Test Q-Learning finds valid path."""
    network = NetworkGraph.create_mesh(num_nodes=5)
    agent = QLearningRouting(network)
    agent.train(source=0, destination=4, num_episodes=500)

    path, cost, time = agent.find_path(0, 4)

    assert len(path) > 0
    assert path[0] == 0
    assert path[-1] == 4
    assert cost > 0
    assert time >= 0

@pytest.fixture
def sample_network():
    """Fixture providing sample network for tests."""
    return NetworkGraph.create_mesh(num_nodes=10)

def test_with_fixture(sample_network):
    """Test using fixture."""
    agent = QLearningRouting(sample_network)
    assert agent.network.num_nodes == 10
```

### Test Coverage

Aim for **80%+ test coverage**:

```bash
# Generate coverage report
pytest --cov=backend --cov=frontend --cov-report=html

# Open report
# Windows:
start htmlcov/index.html
# Linux/Mac:
open htmlcov/index.html
```

---

## 💬 Commit Guidelines

### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only changes
- `style`: Formatting, missing semicolons, etc. (no code change)
- `refactor`: Code restructuring (no feature change)
- `perf`: Performance improvement
- `test`: Adding or updating tests
- `chore`: Maintenance tasks, dependencies

**Scopes:**

- `q-learning`: Q-Learning algorithm
- `dqn`: Deep Q-Network
- `ui`: User interface
- `docs`: Documentation
- `tests`: Testing
- `config`: Configuration

### Examples

**Good commit messages:**

```bash
feat(dqn): add Double DQN implementation

Implemented Double DQN to reduce Q-value overestimation.
Uses policy network for action selection and target network
for value estimation.

Closes #42
```

```bash
fix(q-learning): resolve episode counting bug

Fixed double-counting of episodes when training same network
multiple times. Added tracking of episodes before training starts.

Fixes #58
```

```bash
docs(readme): update installation instructions

- Added conda installation method
- Updated Python version requirement to 3.8+
- Added troubleshooting section
```

```bash
perf(dqn): optimize replay buffer sampling

Replaced list comprehension with numpy indexing for 30% speedup
in batch sampling from replay memory.
```

**Bad commit messages:**

```bash
fix bug
updated code
changes
asdf
WIP
```

### Commit Best Practices

- **One logical change per commit**
- **Write in imperative mood** ("Add feature" not "Added feature")
- **First line < 50 characters**
- **Body line length < 72 characters**
- **Reference issues/PRs** in footer

---

## 📚 Documentation

### When to Update Documentation

Update docs when you:

- ✅ Add new features
- ✅ Change API or function signatures
- ✅ Fix bugs (add to known issues)
- ✅ Change configuration options
- ✅ Add dependencies

### Documentation Structure

```
docs/
├── SETUP.md              # Installation guide
├── USAGE.md              # Usage examples
├── ARCHITECTURE.md       # System design
├── API.md               # Code reference
└── tutorials/           # Step-by-step guides
    ├── custom_networks.md
    ├── training.md
    └── reward_design.md
```

### Code Documentation

**For functions/methods:**

```python
def calculate_reward(
    self,
    current_node: int,
    action: int,
    destination: int,
    visited: Set[int],
    path_length: int
) -> float:
    """
    Calculate reward for taking action in current state.

    Reward components:
    - Positive: Reaching destination, moving closer
    - Negative: Step penalty, loops, wrong direction

    Args:
        current_node: Current position in network
        action: Next node to visit
        destination: Target destination node
        visited: Set of already visited nodes
        path_length: Current path length

    Returns:
        Reward value (typically -10 to +100)

    Example:
        >>> reward = designer.calculate_reward(0, 1, 9, {0}, 1)
        >>> print(reward)  # e.g., 3.5
    """
    pass
```

**For classes:**

```python
class DQNRouting:
    """
    Deep Q-Network based routing with transfer learning.

    Implements DQN algorithm for network routing with:
    - Neural network Q-value approximation
    - Experience replay memory
    - Target network for stability
    - Double DQN for reduced overestimation
    - Transfer learning across networks

    Attributes:
        network: NetworkGraph instance
        config: Configuration object
        policy_net: Main neural network
        target_net: Target network for stable training
        memory: Experience replay buffer
        epsilon: Current exploration rate

    Example:
        >>> network = NetworkGraph.create_mesh(10)
        >>> agent = DQNRouting(network)
        >>> agent.train(0, 9, num_episodes=3000)
        >>> path, cost, time = agent.find_path(0, 9)
    """
    pass
```

---

## 🎨 UI Contributions

### Streamlit Guidelines

When contributing to the UI:

1. **Maintain consistent styling**

   ```python
   st.markdown("### Section Title")  # Use ### for sections
   st.info("ℹ️ Info message")       # Use emojis
   st.success("✅ Success")
   st.warning("⚠️ Warning")
   st.error("❌ Error")
   ```

2. **Add helpful descriptions**

   ```python
   st.slider(
       "Number of Episodes",
       min_value=100,
       max_value=5000,
       value=1000,
       step=100,
       help="More episodes = better convergence but slower training"
   )
   ```

3. **Use progress indicators**
   ```python
   progress_bar = st.progress(0)
   for i in range(episodes):
       # Training logic
       progress_bar.progress((i + 1) / episodes)
   ```

---

## 🌟 Recognition

Contributors will be recognized in:

- **README.md** Contributors section
- **Release notes** for their contributions
- **Project documentation**

---

## 🚀 Release Process

### Version Numbers

We use [Semantic Versioning](https://semver.org/):

- **Major (1.0.0)**: Breaking changes
- **Minor (0.1.0)**: New features (backwards compatible)
- **Patch (0.0.1)**: Bug fixes

### Creating a Release

Maintainers follow this process:

1. Update `CHANGELOG.md`
2. Update version in `setup.py` (if exists)
3. Create git tag: `git tag -a v1.0.0 -m "Release v1.0.0"`
4. Push tag: `git push origin v1.0.0`
5. Create GitHub release
6. Update documentation

---

## ❓ Questions?

If you have questions, you can:

- 💬 Open a [Discussion](https://github.com/shivanshpandeygp/intelligent_routing_system/discussions)
- 📧 Email maintainers: [your.email@example.com]
- 🐛 Check existing [Issues](https://github.com/shivanshpandeygp/intelligent_routing_system/issues)
- 📖 Read the [documentation](docs/)

---

## 📋 Pull Request Template

When you submit a PR, use this template:

```markdown
## Description

Brief description of changes

## Type of Change

- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to not work)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing

Describe the tests you ran and how to reproduce.

## Screenshots (if applicable)

Add screenshots for UI changes.

## Checklist

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added to complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] No merge conflicts
- [ ] Commit messages follow guidelines

## Related Issues

Closes #(issue number)
```

---

## 🙏 Thank You!

Your contributions make this project better! Every contribution, no matter how small, is valued and appreciated.

**Happy Contributing! 🎉**

---

## 📞 Contact

**Project Maintainer:**  
[Shivansh Pandey]  
Email: [sdutt081@gmail.com]  
GitHub: [@shivanshpandeygp](https://github.com/shivanshpandeygp)

**Project Repository:**  
https://github.com/shivanshpandeygp/intelligent_routing_system

---

_Last Updated: November 2025_

---
