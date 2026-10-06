import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import override

import requests
from mlflow.environment_variables import MLFLOW_TRACKING_TOKEN

from cdcr_lexical_diversity_pairwise_scoring import logger


class TrackingAuthorization(ABC):
    """Makes the credentials of the tracking server available to the MLflow client.

    `apply()` is called by the tracker right before every request to the server, so an
    implementation can renew short-lived credentials on the way.
    """

    @abstractmethod
    def apply(self) -> None: ...


class NoAuthorization(TrackingAuthorization):
    """Nothing to do: the server is open, or uses basic auth.

    MLflow reads basic-auth credentials itself from MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD.
    """

    @override
    def apply(self) -> None:
        return


class KeycloakAuthorization(TrackingAuthorization):
    """Bearer tokens of a Keycloak service-account client (OAuth2 client credentials grant).

    The MLflow client reads MLFLOW_TRACKING_TOKEN from the environment on every request, so the
    current token is exported there. Keycloak tokens are short-lived (minutes) while a training run
    takes hours: a new token is fetched once the current one is within `refresh_margin_seconds`
    of its expiry.
    """

    def __init__(
        self,
        token_url: str,
        client_id: str,
        client_secret: str,
        refresh_margin_seconds: float,
        request_timeout_seconds: float,
        session: requests.Session,
        clock: Callable[[], float],
    ) -> None:
        self._token_url = token_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._refresh_margin_seconds = refresh_margin_seconds
        self._request_timeout_seconds = request_timeout_seconds
        self._session = session
        self._clock = clock
        self._token: str | None = None
        self._expires_at = 0.0

    @override
    def apply(self) -> None:
        if self._token is None or self._clock() >= self._expires_at - self._refresh_margin_seconds:
            self._refresh()

        os.environ[MLFLOW_TRACKING_TOKEN.name] = self._token

    def _refresh(self) -> None:
        requested_at = self._clock()
        response = self._session.post(
            self._token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
            },
            timeout=self._request_timeout_seconds,
        )
        response.raise_for_status()

        payload = response.json()
        self._token = payload["access_token"]
        self._expires_at = requested_at + payload["expires_in"]
        logger.debug(
            f"Fetched a tracking token for Keycloak client '{self._client_id}' (expires in {payload['expires_in']} s).",
        )


class AuthorizationFactory:
    """Keycloak authorization when a client id is configured, none otherwise."""

    @staticmethod
    def build(
        token_url: str,
        client_id: str,
        client_secret: str,
        basic_auth_username: str,
        refresh_margin_seconds: float,
        request_timeout_seconds: float,
    ) -> TrackingAuthorization:
        if client_id == "":
            return NoAuthorization()

        if token_url == "":
            raise ValueError("KEYCLOAK_CLIENT_ID is set but KEYCLOAK_TOKEN_URL is empty.")

        # MLflow sends basic auth instead of the bearer token when both are present, so the
        # server would reject every request
        if basic_auth_username != "":
            raise ValueError("Unset MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD to authorize through Keycloak.")

        return KeycloakAuthorization(
            token_url=token_url,
            client_id=client_id,
            client_secret=client_secret,
            refresh_margin_seconds=refresh_margin_seconds,
            request_timeout_seconds=request_timeout_seconds,
            session=requests.Session(),
            clock=time.monotonic,
        )
