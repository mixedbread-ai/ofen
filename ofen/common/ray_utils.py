from __future__ import annotations

import inspect
from functools import wraps
from typing import TYPE_CHECKING, TypeVar

from ofen.common.utils import ensure_import
from ofen.logger import LOGGER

if TYPE_CHECKING:
    import types

# Ensure Ray-related dependencies are imported only when needed
with ensure_import("ofen[ray]"):
    from ray import serve
    from ray.exceptions import RayActorError
    from ray.serve.handle import DeploymentHandle
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


DH = TypeVar("DH", bound=DeploymentHandle)


def remove_remote_from_handle(
    deployment_handle: DH,
    base_module: types.ModuleType,
    max_retries: int = 3,
    min_wait: float = 1,
    max_wait: float = 10,
) -> DH:
    """Enhance the deployment handle with retry logic and async/sync method handling.

    Args:
    ----
        deployment_handle (DH): The deployment handle to enhance.
        base_module (types.ModuleType): The module containing non-remote versions of methods.
        max_retries (int): Maximum number of retry attempts.
        min_wait (float): Minimum wait time between retries in seconds.
        max_wait (float): Maximum wait time between retries in seconds.

    Returns:
    -------
        DH: The enhanced deployment handle.

    """

    class DeploymentProxy:
        def __init__(self):
            self.actor = deployment_handle
            self.retry_decorator = retry(
                stop=stop_after_attempt(max_retries),
                wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
                retry=retry_if_exception_type((RayActorError, Exception)),
            )
            self._wrapped_methods = {}

        def __getattr__(self, name):
            if name.startswith("__") and name.endswith("__"):
                return super().__getattribute__(name)

            if name not in self._wrapped_methods:
                remote_func = getattr(self.actor, name)
                base_func = getattr(base_module, name, None)

                if not callable(base_func):
                    return remote_func

                if inspect.iscoroutinefunction(base_func):

                    @self.retry_decorator
                    @wraps(base_func)
                    async def remote_wrapper_async(*args, **kwargs):
                        try:
                            return await remote_func.remote(*args, **kwargs)
                        except Exception as e:
                            LOGGER.error(f"Error in async remote call: {e!s}")
                            raise

                    self._wrapped_methods[name] = remote_wrapper_async
                else:

                    @self.retry_decorator
                    @wraps(base_func)
                    def remote_wrapper(*args, **kwargs):
                        try:
                            return remote_func.remote(*args, **kwargs)
                        except Exception as e:
                            LOGGER.error(f"Error in sync remote call: {e!s}")
                            raise

                    self._wrapped_methods[name] = remote_wrapper

            return self._wrapped_methods[name]

    return DeploymentProxy()


def normalize_name(name: str) -> str:
    """Normalize the name to be a valid URL path.

    Args:
    ----
        name (str): The name to normalize.

    Returns:
    -------
        str: The normalized name.

    """
    return name.lower().replace(" ", "-").replace("_", "-").replace("/", "_")


def get_deployment(deployment_name: str, app_name: str | None = None) -> DeploymentHandle | None:
    """Get an existing Ray deployment.

    Args:
    ----
        deployment_name (str): The name of the deployment.
        app_name (Optional[str]): The name of the app. If None, uses the normalized deployment name.

    Returns:
    -------
        Optional[DeploymentHandle]: The deployment handle if found, None otherwise.

    """
    normalized_name = normalize_name(deployment_name)
    app_name = normalize_name(app_name) if app_name else normalized_name
    try:
        return serve.get_deployment_handle(normalized_name, app_name=app_name)
    except (KeyError, ValueError, serve.exceptions.RayServeException):
        return None


def create_deployment(name: str, deployment_cls: type, **kwargs) -> serve.Deployment:
    """Create a new Ray deployment.

    Args:
    ----
        name (str): The name of the deployment.
        deployment_cls (Type): The class to deploy.
        **kwargs: Additional arguments for the deployment.

    Returns:
    -------
        serve.Deployment: The created deployment.

    """
    normalized_name = normalize_name(name)
    return serve.deployment(deployment_cls, name=normalized_name, **kwargs)


def run_deployment(
    deployment: serve.Application, name_or_path: str, run_kwargs: dict, *args, **kwargs
) -> serve.Application:
    """Run a deployment.

    Args:
    ----
        deployment (serve.Application): The deployment to run.
        name_or_path (str): The name or path of the deployment.
        run_kwargs (dict): Additional keyword arguments for serve.run.
        *args: Additional arguments for the deployment.
        **kwargs: Additional keyword arguments for the deployment.

    Returns:
    -------
        serve.Application: The running application.

    """
    normalized_name = normalize_name(name_or_path)
    return serve.run(
        deployment.bind(name_or_path, *args, **kwargs),
        **{"name": normalized_name, "route_prefix": f"/{normalized_name}", **run_kwargs},
    )


def delete_deployment(name: str) -> None:
    """Delete an existing Ray deployment.

    Args:
    ----
        name (str): The name of the deployment to delete.

    """
    normalized_name = normalize_name(name)
    try:
        serve.delete(name=normalized_name)
        LOGGER.info(f"Deployment '{normalized_name}' deleted successfully.")
    except Exception as e:
        LOGGER.error(f"Failed to delete deployment '{normalized_name}': {e!s}")
        raise
