"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


class Registry:
    mapping = {
        "builder_name_mapping": {},
        "task_name_mapping": {},
        "processor_name_mapping": {},
        "model_name_mapping": {},
        "lr_scheduler_name_mapping": {},
        "runner_name_mapping": {},
        "state": {},
        "paths": {},
    }

    @classmethod
    def register_builder(cls, name):
        r"""Register a dataset builder to registry with key 'name'

        Args:
            name: Key with which the builder will be registered.

        Usage:

            from minigpt4.common.registry import registry
            from minigpt4.datasets.base_dataset_builder import BaseDatasetBuilder
        """

        def wrap(builder_cls):
            from minigpt4.datasets.builders.base_dataset_builder import BaseDatasetBuilder

            assert issubclass(
                builder_cls, BaseDatasetBuilder
            ), "All builders must inherit BaseDatasetBuilder class, found {}".format(
                builder_cls
            )
            if name in cls.mapping["builder_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["builder_name_mapping"][name]
                    )
                )
            cls.mapping["builder_name_mapping"][name] = builder_cls
            return builder_cls

        return wrap

    @classmethod
    def register_task(cls, name):
        r"""Register a task to registry with key 'name'

        Args:
            name: Key with which the task will be registered.

        Usage:

            from minigpt4.common.registry import registry
        """

        def wrap(task_cls):
            from minigpt4.tasks.base_task import BaseTask

            assert issubclass(
                task_cls, BaseTask
            ), "All tasks must inherit BaseTask class"
            if name in cls.mapping["task_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["task_name_mapping"][name]
                    )
                )
            cls.mapping["task_name_mapping"][name] = task_cls
            return task_cls

        return wrap

    @classmethod
    def register_model(cls, name):
        r"""Register a task to registry with key 'name'

        Args:
            name: Key with which the task will be registered.

        Usage:

            from minigpt4.common.registry import registry
        """

        def wrap(model_cls):
            from minigpt4.models import BaseModel

            assert issubclass(
                model_cls, BaseModel
            ), "All models must inherit BaseModel class"
            if name in cls.mapping["model_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["model_name_mapping"][name]
                    )
                )
            cls.mapping["model_name_mapping"][name] = model_cls
            return model_cls

        return wrap

    @classmethod
    def register_processor(cls, name):
        r"""Register a processor to registry with key 'name'

        Args:
            name: Key with which the task will be registered.

        Usage:

            from minigpt4.common.registry import registry
        """

        def wrap(processor_cls):
            from minigpt4.processors import BaseProcessor

            assert issubclass(
                processor_cls, BaseProcessor
            ), "All processors must inherit BaseProcessor class"
            if name in cls.mapping["processor_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["processor_name_mapping"][name]
                    )
                )
            cls.mapping["processor_name_mapping"][name] = processor_cls
            return processor_cls

        return wrap

    @classmethod
    def register_lr_scheduler(cls, name):
        r"""Register a model to registry with key 'name'

        Args:
            name: Key with which the task will be registered.

        Usage:

            from minigpt4.common.registry import registry
        """

        def wrap(lr_sched_cls):
            if name in cls.mapping["lr_scheduler_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["lr_scheduler_name_mapping"][name]
                    )
                )
            cls.mapping["lr_scheduler_name_mapping"][name] = lr_sched_cls
            return lr_sched_cls

        return wrap

    @classmethod
    def register_runner(cls, name):
        r"""Register a model to registry with key 'name'

        Args:
            name: Key with which the task will be registered.

        Usage:

            from minigpt4.common.registry import registry
        """

        def wrap(runner_cls):
            if name in cls.mapping["runner_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["runner_name_mapping"][name]
                    )
                )
            cls.mapping["runner_name_mapping"][name] = runner_cls
            return runner_cls

        return wrap

    @classmethod
    def register_path(cls, name, path):
        r"""Register a path to registry with key 'name'

        Args:
            name: Key with which the path will be registered.

        Usage:

            from minigpt4.common.registry import registry
        """
        assert isinstance(path, str), "All path must be str."
        if name in cls.mapping["paths"]:
            raise KeyError("Name '{}' already registered.".format(name))
        cls.mapping["paths"][name] = path

    @classmethod
    def register(cls, name, obj):
        r"""Register an item to registry with key 'name'

        Args:
            name: Key with which the item will be registered.

        Usage::

            from minigpt4.common.registry import registry

            registry.register("config", {})
        """
        path = name.split(".")
        current = cls.mapping["state"]

        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[path[-1]] = obj

    # @classmethod
    # def get_trainer_class(cls, name):
    #     return cls.mapping["trainer_name_mapping"].get(name, None)

    @classmethod
    def get_builder_class(cls, name):
        return cls.mapping["builder_name_mapping"].get(name, None)

    @classmethod
    def get_model_class(cls, name):
        return cls.mapping["model_name_mapping"].get(name, None)

    @classmethod
    def get_task_class(cls, name):
        return cls.mapping["task_name_mapping"].get(name, None)

    @classmethod
    def get_processor_class(cls, name):
        return cls.mapping["processor_name_mapping"].get(name, None)

    @classmethod
    def get_lr_scheduler_class(cls, name):
        return cls.mapping["lr_scheduler_name_mapping"].get(name, None)

    @classmethod
    def get_runner_class(cls, name):
        return cls.mapping["runner_name_mapping"].get(name, None)

    @classmethod
    def list_runners(cls):
        return sorted(cls.mapping["runner_name_mapping"].keys())

    @classmethod
    def list_models(cls):
        return sorted(cls.mapping["model_name_mapping"].keys())

    @classmethod
    def list_tasks(cls):
        return sorted(cls.mapping["task_name_mapping"].keys())

    @classmethod
    def list_processors(cls):
        return sorted(cls.mapping["processor_name_mapping"].keys())

    @classmethod
    def list_lr_schedulers(cls):
        return sorted(cls.mapping["lr_scheduler_name_mapping"].keys())

    @classmethod
    def list_datasets(cls):
        return sorted(cls.mapping["builder_name_mapping"].keys())

    @classmethod
    def get_path(cls, name):
        return cls.mapping["paths"].get(name, None)

    @classmethod
    def get(cls, name, default=None, no_warning=False):
        r"""Get an item from registry with key 'name'

        Args:
            name (string): Key whose value needs to be retrieved.
            default: If passed and key is not in registry, default value will
                     be returned with a warning. Default: None
            no_warning (bool): If passed as True, warning when key doesn't exist
                               will not be generated. Useful for MMF's
                               internal operations. Default: False
        """
        original_name = name
        name = name.split(".")
        value = cls.mapping["state"]
        for subname in name:
            value = value.get(subname, default)
            if value is default:
                break

        if (
            "writer" in cls.mapping["state"]
            and value == default
            and no_warning is False
        ):
            cls.mapping["state"]["writer"].warning(
                "Key {} is not present in registry, returning default value "
                "of {}".format(original_name, default)
            )
        return value

    @classmethod
    def unregister(cls, name):
        r"""Remove an item from registry with key 'name'

        Args:
            name: Key which needs to be removed.
        Usage::

            from mmf.common.registry import registry

            config = registry.unregister("config")
        """
        return cls.mapping["state"].pop(name, None)


registry = Registry()
