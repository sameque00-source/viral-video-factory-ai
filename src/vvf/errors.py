"""Exceções do projeto. `retryable=False` impede novas tentativas no executor de etapas."""


class VVFError(Exception):
    retryable = True


class UserInputError(VVFError):
    """Entrada inválida do usuário (arquivo, flag, plano). Repetir não adianta."""

    retryable = False


class IngestError(UserInputError):
    pass


class PlanError(UserInputError):
    pass


class ConsentRequired(UserInputError):
    pass


class PaidCallsDisabled(UserInputError):
    pass


class BudgetExceeded(UserInputError):
    pass


class UnknownPrice(UserInputError):
    pass


class MissingCredential(UserInputError):
    pass


class ProviderError(VVFError):
    """Falha do provedor externo (rede, fila, saída inválida). Pode valer repetir."""


class StepFailed(VVFError):
    retryable = False


class RenderVerificationError(VVFError):
    """O arquivo renderizado não passou nas verificações; repetir o mesmo comando não ajuda."""

    retryable = False
