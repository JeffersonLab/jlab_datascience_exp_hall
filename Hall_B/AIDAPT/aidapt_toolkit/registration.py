import importlib
import logging

aidapt_log = logging.getLogger("AIDAPT Registry")


def load(name):
    mod_name, attr_name = name.split(":")
    aidapt_log.info(f"Attempting to load {mod_name} with {attr_name}")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


class AIDAPTSpec(object):
    def __init__(self, id, entry_point=None, kwargs=None):
        self.id = id
        self.entry_point = entry_point
        self._kwargs = {} if kwargs is None else kwargs

    def make(self, **kwargs):
        """Instantiates an instance of an AIDAPT module with appropriate kwargs"""
        if self.entry_point is None:
            aidapt_log.error(
                "Attempting to make deprecated module {}. \
                               (HINT: is there a newer registered version \
                               of this module?)".format(
                    self.id
                )
            )
            raise
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        if callable(self.entry_point):
            gen = self.entry_point(**_kwargs)
        else:
            cls = load(self.entry_point)
            gen = cls(**_kwargs)

        return gen


class AIDAPTRegistry(object):
    def __init__(self):
        self.aidapt_specs = {}

    def make(self, path, **kwargs):
        if len(kwargs) > 0:
            aidapt_log.info("Making new module: %s (%s)", path, kwargs)
        else:
            aidapt_log.info("Making new module: %s", path)
        aidapt_spec = self.spec(path)
        unfolding = aidapt_spec.make(**kwargs)

        return unfolding

    def all(self):
        return self.aidapt_specs.values()

    def spec(self, path):
        if ":" in path:
            mod_name, _sep, id = path.partition(":")
            try:
                importlib.import_module(mod_name)
            except ImportError:
                aidapt_log.error(
                    "A module ({}) was specified but was not found, \
                                   make sure the package is installed with `pip install` before \
                                   calling `aidapt_module.make()`".format(
                        mod_name
                    )
                )
                raise

        else:
            id = path

        try:
            return self.aidapt_specs[id]
        except KeyError:
            aidapt_log.error("No registered module with id: {}".format(id))
            raise

    def register(self, id, **kwargs):
        if id in self.aidapt_specs:
            aidapt_log.error("Cannot re-register id: {}".format(id))
            raise
        self.aidapt_specs[id] = AIDAPTSpec(id, **kwargs)


# Global unfolding registry
aidapt_registry = AIDAPTRegistry()


def register(id, **kwargs):
    return aidapt_registry.register(id, **kwargs)


def make(id, **kwargs):
    return aidapt_registry.make(id, **kwargs)


def spec(id):
    return aidapt_registry.spec(id)


def list_registered_modules():
    return list(aidapt_registry.aidapt_specs.keys())
