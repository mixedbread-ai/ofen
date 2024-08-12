from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable

from ofen.common.ray_utils import (
    create_deployment,
    delete_deployment,
    get_deployment,
    remove_remote_from_handle,
    run_deployment,
)
from ofen.logger import LOGGER
from ofen.runners.runner import Runner, T

if TYPE_CHECKING:
    from ofen.batch_processor.batch_processor import BatchProcessorConfig
    from ofen.models.base.model import BaseModel


def make_ray_runner(
    version: str = "0.0.1",
    batch_config: BatchProcessorConfig | None = None,
    deployment_kwargs: dict | None = None,
    run_kwargs: dict | None = None,
    bind_kwargs: dict | None = None,
) -> Callable[[type[BaseModel]], type[BaseModel]]:
    """Decorator factory for creating a Ray-based engine class.

    Args:
    ----
        version (str): Version of the engine.
        batch_config (Optional[BatchProcessorConfig]): Configuration for batch processing.
        deployment_kwargs (Optional[dict]): Additional kwargs for deployment creation.
        run_kwargs (Optional[dict]): Additional kwargs for serve.run.
        bind_kwargs (Optional[dict]): Additional kwargs for deployment.bind.

    Returns:
    -------
        Callable[[Type[BaseModel]], Type[BaseModel]]: A decorator function that creates a Ray-based engine class.

    """

    def decorator(model_cls: type[BaseModel]) -> type[BaseModel]:
        if inspect.isclass(model_cls):
            return RayRunner.from_model_cls(
                model_cls,
                version=version,
                batch_config=batch_config,
                deployment_kwargs=deployment_kwargs or {},
                run_kwargs=run_kwargs or {},
                bind_kwargs=bind_kwargs or {},
            )
        msg = f"make_ray_engine expects a class, got {type(model_cls)}"
        raise TypeError(msg)

    return decorator


DEFAULT_DEPLOYMENT_OPTIONS = {
    "max_ongoing_requests": 1000,
    "num_replicas": 2,
    "ray_actor_options": {
        "num_cpus": 1,
    },
}


class RayRunner(Runner):
    @classmethod
    def from_model_cls(
        cls,
        model_cls: type[T],
        *,
        version: str | None = None,
        force_redeploy: bool = False,
        run_kwargs: dict | None = None,
        deployment_kwargs: dict | None = None,
        **kwargs,
    ) -> type[T]:
        """Create a Ray-based engine class from a model class.

        Args:
        ----
            model_cls (Type[T]): The base model class.
            version (Optional[str]): Version of the engine.
            force_redeploy (bool): Whether to force redeployment.
            run_kwargs (dict): Additional kwargs for serve.run.
            deployment_kwargs (dict): Additional kwargs for deployment creation.
                See: https://docs.ray.io/en/latest/serve/api/doc/ray.serve.Deployment.html
            **kwargs: Additional arguments passed to the parent class.

        Returns:
        -------
            Type[T]: A derived class with Ray deployment capabilities.

        """
        if deployment_kwargs is None:
            deployment_kwargs = {}
        if run_kwargs is None:
            run_kwargs = {}
        engine_cls = super().from_model_cls(model_cls, **kwargs)

        class RayWrapper:
            def __init__(self, name_or_path: str, *args: Any, **kwargs: Any) -> None:
                self.name_or_path = name_or_path

                merged_run_kwargs = {**run_kwargs, **kwargs.pop("run_kwargs", {})}
                merged_deployment_kwargs = {
                    **DEFAULT_DEPLOYMENT_OPTIONS,
                    **deployment_kwargs,
                    **kwargs.pop("deployment_kwargs", {}),
                }

                deployment_handle = get_deployment(name_or_path, app_name=name_or_path)
                if not deployment_handle or force_redeploy:
                    LOGGER.info(f"Creating deployment for {name_or_path}")
                    deployment = create_deployment(name_or_path, engine_cls, **merged_deployment_kwargs)
                    deployment_handle = run_deployment(deployment, name_or_path, merged_run_kwargs, *args, **kwargs)
                self._remote_handle = remove_remote_from_handle(deployment_handle, base_module=engine_cls)

            @property
            def __class__(self) -> type[T]:
                return engine_cls

            def __getattr__(self, name: str) -> Any:
                return getattr(self._remote_handle, name)

            def __call__(self, *args: Any, **kwargs: Any) -> None:
                self._remote_handle.remote(*args, **kwargs)

            def delete_deployment(self) -> None:
                try:
                    delete_deployment(self.name_or_path)
                except Exception as e:
                    LOGGER.warning(f"Failed to delete deployment: {e}")

        class DerivedClass(engine_cls):
            def __new__(cls, name_or_path: str, *args: Any, **kwargs: Any) -> RayWrapper:
                return RayWrapper(name_or_path, *args, **kwargs)

        return DerivedClass
