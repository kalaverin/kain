from kain.internals import Who

__all__ = 'Missing', 'Nothing', 'Singleton'


class Singleton(type):

    def __init__(cls, name, parents, attrbutes):
        super().__init__(name, parents, attrbutes)

    def __call__(cls, *args, **kw):
        if not hasattr(cls, 'instance'):
            cls.instance = super().__call__(*args, **kw)
        return cls.instance


class Missing:

    def __hash__(self):
        return id(self)

    def __bool__(self):
        return False

    def __eq__(self, _):
        return False

    def __repr__(self):
        return f'<{Who.Name(self, addr=True)}>'


Nothing = Missing()
