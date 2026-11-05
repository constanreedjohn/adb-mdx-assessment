"""Database URL building (no database needed)."""

from pydantic import SecretStr

from mdx_assessment.core.persistence.db import database_url
from mdx_assessment.secret_store import MissingSecretError, SecretName
from mdx_assessment.settings import PostgresSettings


class FakeSecrets:
    backend = "fake"

    def __init__(self, password: str | None) -> None:
        self._password = password

    def get(self, name: SecretName) -> SecretStr | None:
        return SecretStr(self._password) if self._password else None

    def require(self, name: SecretName) -> SecretStr:
        value = self.get(name)
        if value is None:
            raise MissingSecretError(name, self.backend)
        return value


PG = PostgresSettings(host="db.local", port=5433, db="mdx_assessment", user="mdx")


def test_password_with_special_characters_is_escaped() -> None:
    url = database_url(PG, FakeSecrets("p@ss/w:rd%"), database="mdx_assessment_test")
    assert url.password == "p@ss/w:rd%"
    assert url.host == "db.local"
    assert url.port == 5433
    assert url.database == "mdx_assessment_test"
    rendered = url.render_as_string(hide_password=False)
    assert "p%40ss%2Fw%3Ard%25@db.local" in rendered


def test_password_is_hidden_when_printed() -> None:
    url = database_url(PG, FakeSecrets("secret-pw"))
    assert "secret-pw" not in str(url)
    assert "secret-pw" not in repr(url)


def test_missing_password_is_a_clear_error() -> None:
    import pytest

    with pytest.raises(MissingSecretError, match="postgres-password"):
        database_url(PG, FakeSecrets(None))
