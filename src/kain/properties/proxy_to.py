from functools import partial
from logging import getLogger
from operator import attrgetter

from kain.classes import Missing
from kain.internals import (
    Is,
    Who,
    get_attr,
)
from kain.properties.primitives import bound_property

__all__ = ("proxy_to",)

Nothing = Missing()
logger = getLogger(__name__)


def proxy_to(  # noqa: PLR0915
    *mapping,
    getter=attrgetter,
    default=Nothing,
    pre=None,
    safe=True,
):
    if isinstance(mapping[-1], str):
        bind = bound_property

    elif mapping[-1] is None:
        bind, mapping = None, mapping[:-1]

    else:
        bind, mapping = mapping[-1], mapping[:-1]

    def class_wraper(cls):  # noqa: PLR0915
        if not Is.Class(cls):
            msg = f"{Who.Is(cls)} isn't a class"
            raise TypeError(msg)

        try:
            fields = cls.__proxy_fields__
        except AttributeError:
            fields = []
            cls.__proxy_fields__ = fields

        pivot, mapping_list = mapping[0], mapping[1:]

        if not mapping_list or (
            len(mapping_list) == 1 and not isinstance(mapping_list[0], str)
        ):
            raise ValueError(f"empty {mapping_list=} for {pivot=}")

        for method in mapping_list:

            if safe and not method.startswith("_") and get_attr(cls, method):
                msg = f"{Who.Is(cls)} already exists {method!a}: {get_attr(cls, method)}"
                raise TypeError(msg)

            def wrapper(name, node):
                if not isinstance(pivot, str):
                    try:
                        return getattr(pivot, name)
                    except AttributeError as e:
                        msg = (
                            f"{Who.Is(node)}.{name} {Who.Name(getter)[:4]}-proxied -> "
                            f"{Who.Is(pivot)}.{name}, but last isn't exists"
                        )
                        raise AttributeError(msg) from e

                try:
                    entity = getattr(node, pivot)
                except AttributeError as e:
                    msg = (
                        f"{Who.Is(node)}.{name} {Who.Name(getter)[:4]}-proxied -> "
                        f"{Who.Is(node)}.{pivot}.{name}, but "
                        f"{Who.Is(node)}.{pivot} isn't exists"
                    )
                    raise AttributeError(msg) from e

                if entity is None:
                    msg = (
                        f"{Who.Is(node)}.{name} {Who.Name(getter)[:4]}-proxied -> "
                        f"{Who.Is(node)}.{pivot}.{name}, but current "
                        f"{Who.Is(node)}.{pivot} is None"
                    )

                    if default is Nothing:
                        raise AttributeError(msg)

                    msg = f"{msg}; return {Who.Is(default)}"
                    logger.warning(msg)
                    result = default

                else:
                    try:
                        result = getter(name)(entity)

                    except (AttributeError, KeyError) as e:
                        msg = (
                            f"{Who.Is(node)}.{name} {Who.Name(getter)[:4]}-proxied -> "
                            f"{Who.Is(node)}.{pivot}.{name}, but isn't exists "
                            f"('{name}' not in {Who.Is(node)}.{pivot}): "
                            f"{Who.Is(entity)}"
                        )

                        if default is Nothing:
                            raise Is.classOf(e)(msg) from e

                        msg = f"{msg}; return {Who.Is(default)}"
                        logger.warning(msg)
                        result = default

                return partial(pre, result) if pre else result

            wrapper.__name__ = method
            wrapper.__qualname__ = f"{pivot}.{method}"

            if bind is None:
                node = cls.__dict__[pivot]
                try:
                    value = node.__dict__[method]
                except KeyError:
                    value = getattr(node, method)
            else:
                wrap = partial(wrapper, method)
                wrap.__name__ = method
                wrap.__qualname__ = f"{pivot}.{method}"
                value = bind(wrap)

            fields.append(method)
            setattr(cls, method, value)
            cls.__proxy_fields__.sort()

        return cls

    return class_wraper
