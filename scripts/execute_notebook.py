"""Execute a notebook in place using the installed Jupyter kernel client."""

from __future__ import annotations

import argparse
from pathlib import Path
from queue import Empty

import nbformat
from jupyter_client import KernelManager


def execute_notebook(path: Path, cwd: Path, timeout: int = 600) -> None:
    notebook = nbformat.read(path, as_version=4)
    kernel_name = notebook.metadata.get("kernelspec", {}).get("name", "python3")
    manager = KernelManager(kernel_name=kernel_name)
    manager.start_kernel(cwd=str(cwd))
    client = manager.client()
    client.start_channels()
    execution_error: RuntimeError | None = None

    try:
        client.wait_for_ready(timeout=timeout)
        for cell_number, cell in enumerate(notebook.cells, start=1):
            if cell.cell_type != "code":
                continue
            cell.outputs = []
            message_id = client.execute(cell.source, stop_on_error=True)
            while True:
                try:
                    message = client.get_iopub_msg(timeout=timeout)
                except Empty as exc:
                    raise TimeoutError(
                        f"Notebook cell {cell_number} exceeded {timeout} seconds"
                    ) from exc
                if message.get("parent_header", {}).get("msg_id") != message_id:
                    continue
                message_type = message["msg_type"]
                content = message["content"]
                if message_type == "execute_input":
                    cell.execution_count = content.get("execution_count")
                elif message_type == "stream":
                    cell.outputs.append(
                        nbformat.v4.new_output(
                            "stream", name=content["name"], text=content["text"]
                        )
                    )
                elif message_type in {"display_data", "update_display_data"}:
                    cell.outputs.append(
                        nbformat.v4.new_output(
                            "display_data",
                            data=content.get("data", {}),
                            metadata=content.get("metadata", {}),
                        )
                    )
                elif message_type == "execute_result":
                    cell.outputs.append(
                        nbformat.v4.new_output(
                            "execute_result",
                            data=content.get("data", {}),
                            metadata=content.get("metadata", {}),
                            execution_count=content.get("execution_count"),
                        )
                    )
                elif message_type == "clear_output":
                    cell.outputs = []
                elif message_type == "error":
                    cell.outputs.append(
                        nbformat.v4.new_output(
                            "error",
                            ename=content.get("ename", "Error"),
                            evalue=content.get("evalue", ""),
                            traceback=content.get("traceback", []),
                        )
                    )
                    execution_error = RuntimeError(
                        f"Notebook cell {cell_number} failed: "
                        f"{content.get('ename')}: {content.get('evalue')}"
                    )
                elif message_type == "status" and content.get("execution_state") == "idle":
                    break
            if execution_error is not None:
                break
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)
        nbformat.write(notebook, path)

    if execution_error is not None:
        raise execution_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    execute_notebook(args.notebook.resolve(), args.cwd.resolve(), args.timeout)


if __name__ == "__main__":
    main()
