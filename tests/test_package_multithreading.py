from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from types import ModuleType
from typing import Any

from kagglehub.packages import PackageScope, _apply_context_manager_to_module
from tests.fixtures import BaseTestCase

from .utils import clear_imported_kaggle_packages


class TestPackageMultithreading(BaseTestCase):

    def tearDown(self) -> None:
        clear_imported_kaggle_packages()

    def test_multithreaded_package_sequential(self) -> None:
        # Create our fake module and do the kagglehub import logic
        module = import_fake_package_module("test", package_act)

        thread_a_entered = Event()
        thread_b_entered = Event()
        thread_a_ready_for_exit = Event()
        thread_b_ready_for_exit = Event()

        def thread_a_worker() -> None:
            module.package_act(thread_a_entered, thread_a_ready_for_exit)

        def thread_b_worker() -> None:
            module.package_act(thread_b_entered, thread_b_ready_for_exit)

        thread_a = Thread(target=thread_a_worker)
        thread_b = Thread(target=thread_b_worker)

        # Thread A starts and finishes first
        thread_a.start()
        wait_or_raise(thread_a_entered)
        thread_a_ready_for_exit.set()
        thread_a.join(timeout=3)
        # Then Thread B starts and finishes afterwards
        thread_b.start()
        wait_or_raise(thread_b_entered)
        thread_b_ready_for_exit.set()
        thread_b.join(timeout=3)

    def test_multithreaded_package_interleaving_filo(self) -> None:
        # Create our fake module and do the kagglehub import logic
        module = import_fake_package_module("test", package_act)

        thread_a_entered = Event()
        thread_b_entered = Event()
        thread_a_ready_for_exit = Event()
        thread_b_ready_for_exit = Event()

        def thread_a_worker() -> None:
            module.package_act(thread_a_entered, thread_a_ready_for_exit)

        def thread_b_worker() -> None:
            module.package_act(thread_b_entered, thread_b_ready_for_exit)

        thread_a = Thread(target=thread_a_worker)
        thread_b = Thread(target=thread_b_worker)

        # Thread A starts first
        thread_a.start()
        wait_or_raise(thread_a_entered)
        # Then Thread B starts and finishes
        thread_b.start()
        wait_or_raise(thread_b_entered)
        thread_b_ready_for_exit.set()
        thread_b.join(timeout=3)
        # Then Thread A finally finishes
        thread_a_ready_for_exit.set()
        thread_a.join(timeout=3)

    def test_multithreaded_package_interleaving_fifo(self) -> None:
        # Create our fake module and do the kagglehub import logic
        module = import_fake_package_module("test", package_act)

        thread_a_entered = Event()
        thread_b_entered = Event()
        thread_a_ready_for_exit = Event()
        thread_b_ready_for_exit = Event()

        def thread_a_worker() -> None:
            module.package_act(thread_a_entered, thread_a_ready_for_exit)

        def thread_b_worker() -> None:
            module.package_act(thread_b_entered, thread_b_ready_for_exit)

        thread_a = Thread(target=thread_a_worker)
        thread_b = Thread(target=thread_b_worker)

        # Thread A starts first
        thread_a.start()
        wait_or_raise(thread_a_entered)
        # Then Thread B starts
        thread_b.start()
        wait_or_raise(thread_b_entered)
        # Then Thread A finishes
        thread_a_ready_for_exit.set()
        thread_a.join(timeout=3)
        # Then Thread B finishes
        thread_b_ready_for_exit.set()
        thread_b.join(timeout=3)


def import_fake_package_module(module_name_suffix: str, *members: Callable[..., Any] | type[Any]) -> ModuleType:
    """Makes a fake python module with our custom context manager decorations applied.

    Creates the module with the desired function and class members, and calls _apply_context_manager_to_module
    to decorate those members with our context manager scope logic."""
    module = ModuleType(f"kagglehub_package_{module_name_suffix}")
    setattr(module, "__package_version__", "0.1.0")  # noqa: B010

    # Make a fake dir with the required kagglehub_requirements file.
    # Needs to survive until we init PackageScope which reads it.
    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        module.__file__ = str(temp_dir_path / "__init__.py")
        (temp_dir_path / "kagglehub_requirements.yaml").write_text("format_version: 0.1.0\ndatasources: []")

        # Link all desired members to the module
        for member in members:
            setattr(module, member.__name__, member)
            member.__module__ = module.__name__

        # Do the kagglehub side of the module import
        with PackageScope(module) as package_scope:
            _apply_context_manager_to_module(module, package_scope)

    return module


def package_act(on_enter: Event, ready_for_exit: Event) -> None:
    if not PackageScope.get():
        raise ValueError()

    on_enter.set()

    if not ready_for_exit.wait(1):
        raise TimeoutError()


def wait_or_raise(event: Event, timeout: float = 1) -> None:
    if not event.wait(timeout):
        raise TimeoutError()
