"""AC-7: python_exec — чтение /etc/passwd, сеть, запись и опасные конструкции падают; вычисления работают."""

from __future__ import annotations

import pytest

from reflexdesk.sandbox.executor import SubprocessRunner, execute
from reflexdesk.sandbox.policy import PolicyViolation, check


@pytest.fixture(scope="module")
def runner():
    return SubprocessRunner()


@pytest.mark.parametrize("code", [
    "print(open('/etc/passwd').read())",
    "import socket",
    "import os\nos.system('id')",
    "from subprocess import run",
    "__import__('os')",
    "x = ().__class__.__mro__",
    "eval('1+1')",
    "getattr(object, 'x')",
])
def test_policy_rejects(code):
    with pytest.raises(PolicyViolation):
        check(code)


def test_policy_allows_whitelist():
    check("import numpy as np\nimport pandas as pd\nfrom scipy import stats\nimport math\nresult = math.pi")


def test_computation_succeeds(runner):
    result = execute("import numpy as np\nimport pandas as pd\nprint(pd.Series([1, 2, 3]).mean())\nresult = float(np.sum([1, 2]))", runner)
    assert result.error is None and result.result == 3.0 and "2.0" in result.stdout


@pytest.mark.parametrize("code", [
    "import pandas as pd\nresult = pd.read_csv('/etc/passwd')",
    "import numpy as np\nresult = np.loadtxt('/etc/passwd', dtype=str)",
    "import pandas as pd\nresult = pd.read_csv('http://example.com/data.csv')",
    "import numpy as np\nnp.save('stolen.npy', np.zeros(3))",
])
def test_runtime_guard_blocks_file_and_network(runner, code):
    result = execute(code, runner, timeout_s=30)
    assert result.error is not None and "PermissionError" in result.error


def test_direct_open_blocked(runner):
    result = execute("print(open('/etc/passwd').read())", runner)
    assert result.error.startswith("PolicyViolation")
