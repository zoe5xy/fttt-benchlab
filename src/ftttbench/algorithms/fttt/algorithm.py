"""Lazy public entry: optional neural dependencies are not imported at discovery."""


class FTTTAlgorithm:
    name = "fttt"

    def __init__(self, name="fttt"):
        self.name = name

    def run(self, case, config):
        from ftttbench.core.config import prepare_config
        from ftttbench.core.schema import FTTTCase

        case = FTTTCase.model_validate(case.model_dump())
        config = prepare_config(self.name, config, case=case)
        try:
            from .adapter import reconstruct
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                raise RuntimeError(
                    "FTTT requires PyTorch; install fttt-benchlab[neural]."
                ) from exc
            raise
        result = reconstruct(case, config)
        result.algorithm = self.name
        return result
