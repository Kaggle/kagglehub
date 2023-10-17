from kagglehub import http_resolver, registry


def _install_resolvers():
    registry.resolver.add_implementation(http_resolver.HttpResolver())
    # TODO(b/305947763): Implement Kaggle Cache resolver.
