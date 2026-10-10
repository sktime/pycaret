"""Module to test time_series forecasting utils"""

import pytest

from pycaret.utils.time_series import TSExogenousPresent
from pycaret.utils.time_series.forecasting import _check_and_clean_coverage
from pycaret.utils.time_series.forecasting.models import _disable_exogenous_enforcement

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

##############################
# Functions Start Here ####
##############################

# NOTE: Fixtures can not be used to parameterize tests
# https://stackoverflow.com/questions/52764279/pytest-how-to-parametrize-a-test-with-a-list-that-is-returned-from-a-fixture
# Hence, we have to create functions and create the parameterized list first
# (must happen during collect phase) before passing it to mark.parameterize.

############################
# Functions End Here ####
############################


##########################
# Tests Start Here ####
##########################


def test_check_and_clean_coverage():
    """Tests _check_and_clean_coverage"""

    # Tests floating point value ----
    coverage = 0.9
    coverage = _check_and_clean_coverage(coverage=coverage)
    coverage = [round(value, 2) for value in coverage]
    assert isinstance(coverage, list)
    assert coverage == [0.05, 0.95]

    # Tests List values (sorted) ----
    coverage_expected = [0.1, 0.9]
    coverage = _check_and_clean_coverage(coverage=coverage_expected)
    assert isinstance(coverage, list)
    assert coverage == coverage_expected

    # Tests List values (unsorted) ----
    coverage = [0.9, 0.1]
    coverage = _check_and_clean_coverage(coverage=coverage)
    assert isinstance(coverage, list)
    assert coverage == coverage_expected

    # Tests List values (incorrect length 1) ----
    with pytest.raises(ValueError) as errmsg:
        coverage = [0.1]
        coverage = _check_and_clean_coverage(coverage=coverage)
    exceptionmsg = errmsg.value.args[0]
    assert (
        "When coverage is a list, it must be of length 2 corresponding to"
        in exceptionmsg
    )

    # Tests List values (incorrect length 2) ----
    with pytest.raises(ValueError) as errmsg:
        coverage = [0.1, 0.5, 0.9]
        coverage = _check_and_clean_coverage(coverage=coverage)
    exceptionmsg = errmsg.value.args[0]
    assert (
        "When coverage is a list, it must be of length 2 corresponding to"
        in exceptionmsg
    )

    # Tests incorrect types ----
    with pytest.raises(TypeError) as errmsg:
        coverage = None
        coverage = _check_and_clean_coverage(coverage=coverage)
    exceptionmsg = errmsg.value.args[0]
    assert (
        "'coverage' must be of type float or a List of floats of length 2."
        in exceptionmsg
    )


class _CanonicalExogenousForecaster:
    def __init__(self, capability_exogenous):
        self.capability_exogenous = capability_exogenous

    def get_tag(self, tag_name, tag_value_default=None, raise_error=True):
        if tag_name == "capability:exogenous":
            return self.capability_exogenous
        raise AssertionError(f"Unexpected tag lookup: {tag_name}")


class _LegacyExogenousForecaster:
    def __init__(self, ignores_exogeneous_X):
        self.ignores_exogeneous_X = ignores_exogeneous_X

    def get_tag(self, tag_name, tag_value_default=None, raise_error=True):
        if tag_name == "capability:exogenous":
            assert tag_value_default is None
            assert not raise_error
            return tag_value_default
        if tag_name == "ignores-exogeneous-X":
            return self.ignores_exogeneous_X
        raise AssertionError(f"Unexpected tag lookup: {tag_name}")


@pytest.mark.parametrize(
    "forecaster, expected",
    [
        (_CanonicalExogenousForecaster(capability_exogenous=True), False),
        (_CanonicalExogenousForecaster(capability_exogenous=False), True),
        (_LegacyExogenousForecaster(ignores_exogeneous_X=False), False),
        (_LegacyExogenousForecaster(ignores_exogeneous_X=True), True),
    ],
)
def test_disable_exogenous_enforcement_supports_canonical_and_legacy_tags(
    forecaster, expected
):
    """Tests canonical and legacy exogenous-capability tag semantics."""
    assert (
        _disable_exogenous_enforcement(
            forecaster=forecaster,
            enforce_exogenous=True,
            exp_has_exogenous=TSExogenousPresent.YES,
        )
        is expected
    )


def test_disable_exogenous_enforcement_propagates_tag_lookup_errors():
    """Tests failures unrelated to a missing canonical tag are not hidden."""

    class FailingForecaster:
        def get_tag(self, tag_name, tag_value_default=None, raise_error=True):
            raise RuntimeError("unexpected tag lookup failure")

    with pytest.raises(RuntimeError, match="unexpected tag lookup failure"):
        _disable_exogenous_enforcement(
            forecaster=FailingForecaster(),
            enforce_exogenous=True,
            exp_has_exogenous=TSExogenousPresent.YES,
        )
