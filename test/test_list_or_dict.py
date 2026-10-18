# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the backward-compatible location of ListOrDict."""

import unittest
from test import QiskitAlgorithmsTestCase

from qiskit_algorithms.custom_types import ListOrDict
from qiskit_algorithms.list_or_dict import ListOrDict as LegacyListOrDict


class TestListOrDict(QiskitAlgorithmsTestCase):
    """Test the backward-compatible location of ListOrDict."""

    def test_alias(self):
        """Test that the legacy module provides the same type."""
        self.assertIs(LegacyListOrDict, ListOrDict)


if __name__ == "__main__":
    unittest.main()
