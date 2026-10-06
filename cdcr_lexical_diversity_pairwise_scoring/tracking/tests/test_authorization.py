from http import HTTPStatus
from typing import Any

import pytest
import requests
from mlflow.environment_variables import MLFLOW_TRACKING_TOKEN

from cdcr_lexical_diversity_pairwise_scoring.tracking.authorization import (
    AuthorizationFactory,
    KeycloakAuthorization,
    NoAuthorization,
)

TOKEN_URL = "https://auth.example.test/realms/homelab/protocol/openid-connect/token"


class FakeResponse:
    def __init__(
        self,
        payload: dict[str, Any],
        status_code: int,
    ) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= HTTPStatus.BAD_REQUEST:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeSession:
    """Hands out `token-1`, `token-2`, ... and records every token request."""

    def __init__(
        self,
        expires_in: int,
        status_code: int,
    ) -> None:
        self._expires_in = expires_in
        self._status_code = status_code
        self.requests: list[dict[str, Any]] = []

    def post(
        self,
        url: str,
        data: dict[str, str],
        timeout: float,
    ) -> FakeResponse:
        self.requests.append({"url": url, "data": data, "timeout": timeout})
        payload = {"access_token": f"token-{len(self.requests)}", "expires_in": self._expires_in}
        return FakeResponse(payload, self._status_code)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture(autouse=True)
def clean_token_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MLFLOW_TRACKING_TOKEN.name, raising=False)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def build_keycloak_authorization(
    session: FakeSession,
    clock: FakeClock,
) -> KeycloakAuthorization:
    return KeycloakAuthorization(
        token_url=TOKEN_URL,
        client_id="cdcr-pairwise-scoring",
        client_secret="secret",
        refresh_margin_seconds=60,
        request_timeout_seconds=10,
        session=session,
        clock=clock,
    )


def test_first_apply_fetches_a_client_credentials_token_and_exports_it(clock: FakeClock) -> None:
    session = FakeSession(expires_in=300, status_code=200)
    authorization = build_keycloak_authorization(session, clock)

    authorization.apply()

    assert MLFLOW_TRACKING_TOKEN.get() == "token-1"
    assert session.requests == [
        {
            "url": TOKEN_URL,
            "data": {
                "grant_type": "client_credentials",
                "client_id": "cdcr-pairwise-scoring",
                "client_secret": "secret",
            },
            "timeout": 10,
        },
    ]


def test_token_is_reused_while_it_is_valid(clock: FakeClock) -> None:
    session = FakeSession(expires_in=300, status_code=200)
    authorization = build_keycloak_authorization(session, clock)

    authorization.apply()
    clock.now += 200
    authorization.apply()

    assert len(session.requests) == 1
    assert MLFLOW_TRACKING_TOKEN.get() == "token-1"


def test_token_is_refreshed_within_the_margin_before_expiry(clock: FakeClock) -> None:
    session = FakeSession(expires_in=300, status_code=200)
    authorization = build_keycloak_authorization(session, clock)

    authorization.apply()
    clock.now += 241
    authorization.apply()

    assert MLFLOW_TRACKING_TOKEN.get() == "token-2"


def test_rejected_token_request_raises(clock: FakeClock) -> None:
    session = FakeSession(expires_in=300, status_code=401)
    authorization = build_keycloak_authorization(session, clock)

    with pytest.raises(requests.HTTPError):
        authorization.apply()

    assert MLFLOW_TRACKING_TOKEN.get() is None


def test_no_authorization_leaves_the_environment_untouched() -> None:
    NoAuthorization().apply()

    assert MLFLOW_TRACKING_TOKEN.get() is None


def test_factory_without_a_client_id_builds_no_authorization() -> None:
    authorization = AuthorizationFactory.build(
        token_url="",
        client_id="",
        client_secret="",
        basic_auth_username="",
        refresh_margin_seconds=60,
        request_timeout_seconds=10,
    )

    assert isinstance(authorization, NoAuthorization)


def test_factory_with_a_client_id_builds_keycloak_authorization() -> None:
    authorization = AuthorizationFactory.build(
        token_url=TOKEN_URL,
        client_id="cdcr-pairwise-scoring",
        client_secret="secret",
        basic_auth_username="",
        refresh_margin_seconds=60,
        request_timeout_seconds=10,
    )

    assert isinstance(authorization, KeycloakAuthorization)


def test_factory_rejects_a_client_id_without_a_token_url() -> None:
    with pytest.raises(ValueError, match="KEYCLOAK_TOKEN_URL"):
        AuthorizationFactory.build(
            token_url="",
            client_id="cdcr-pairwise-scoring",
            client_secret="secret",
            basic_auth_username="",
            refresh_margin_seconds=60,
            request_timeout_seconds=10,
        )


def test_factory_rejects_keycloak_together_with_basic_auth() -> None:
    # MLflow prefers basic auth over a bearer token, so the token would never be sent
    with pytest.raises(ValueError, match="MLFLOW_TRACKING_USERNAME"):
        AuthorizationFactory.build(
            token_url=TOKEN_URL,
            client_id="cdcr-pairwise-scoring",
            client_secret="secret",
            basic_auth_username="someone",
            refresh_margin_seconds=60,
            request_timeout_seconds=10,
        )
