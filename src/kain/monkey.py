from contextlib import suppress
from functools import wraps
from logging import getLogger
from typing import ClassVar

from kain.importer import required
from kain.internals import Is, Who

logger = getLogger(__name__)


class Monkey:
    mapping: ClassVar[dict] = {}

    @classmethod
    def expect(cls, *exceptions):
        def make_wrapper(func):
            @wraps(func)
            def wrapper(klass, *args, **kw):
                with suppress(exceptions):
                    return func(klass, *args, **kw)

            return classmethod(wrapper)

        return make_wrapper

    @classmethod
    def patch(cls, module, new):

        if isinstance(module, tuple):
            node, name = module

        elif Is.module(module):
            node, name = module, new.__name__

        else:
            path, name = module.rsplit('.', 1)
            try:
                node = required(path)
            except ImportError:
                logger.error(f'{module=} import error')  # noqa: TRY400
                raise

        if getattr(node, name, None) is new:
            return new

        old = required(node, name) if Who(node, full=False) != name else node

        setattr(node, name, new)
        new = getattr(node, name)
        if old is new:
            raise RuntimeError

        cls.mapping[new] = old
        logger.debug(f'{Who(old, addr=True)} -> {Who(new, addr=True)}')
        return new

    @classmethod
    def bind(cls, node, name=None, decorator=None):
        node = required(node) if isinstance(node, str) else node

        def bind(func):
            @wraps(func)
            def wrapper(*args, **kw):
                if decorator is classmethod:
                    return func(node, *args, **kw)
                return func(*args, **kw)

            local = name or func.__name__
            setattr(node, local, wrapper)
            logger.info(f'{Who(node)}.{local} <- {Who(func, addr=True)}')
            return wrapper

        return bind

    @classmethod
    def wrap(cls, node, name=None, decorator=None):
        node = required(node) if isinstance(node, str) else node

        def wrap(func):

            wrapped_name = name or func.__name__
            if Who(node, full=False) != wrapped_name:
                wrapped_func = required(node, wrapped_name)
            else:
                wrapped_func = node

            @wraps(func)
            def wrapper(*args, **kw):
                return func(wrapped_func, *args, **kw)

            logger.info(f'{Who(node)}.{wrapped_name} <- {Who(func, addr=True)}')

            wrapped = decorator(wrapper) if decorator else wrapper
            cls.patch((node, wrapped_name), wrapped)
            return wrapper

        return wrap
