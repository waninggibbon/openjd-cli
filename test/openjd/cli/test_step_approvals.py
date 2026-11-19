# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Tests for STEP_APPROVALS (RFC 0200) support in the CLI: the approval gate of
`openjd run`, and approvals in `openjd check`, `openjd summary` and `openjd run --help`."""

from __future__ import annotations

import io
import json
import signal
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import yaml

from . import run_openjd_cli_main
from openjd.cli._common._approvals import is_approval_only, step_approval, terminal_safe
from openjd.cli._run._local_session._actions import RunTaskAction
from openjd.cli._run._local_session._approval_gate import ApprovalDecision, StepApprovalGate
from openjd.model import create_job, decode_job_template

TEMPLATES_DIR = Path(__file__).parent / "templates"
# Preview -> ReviewPreview (approval-only) -> Final -> Publish, and Independent.
APPROVALS_JOB = str(TEMPLATES_DIR / "step_approvals_job.yaml")
# One Step with an approval, two Tasks, and a message that uses Step.Name/Job.Name.
APPROVAL_WITH_SCRIPT_JOB = str(TEMPLATES_DIR / "step_approval_with_script.yaml")
PROMPT = "Approve Step 'ReviewPreview'? Enter 'y' to approve or 'n' to reject: "


class FakeTerminal(io.StringIO):
    """A stdin that is an interactive terminal, answering with the given text."""

    def __init__(self, answers: str = "") -> None:
        super().__init__(answers)

    def isatty(self) -> bool:
        return True


class NotATerminal(io.StringIO):
    """A stdin that is not a terminal, and that a test fails if anything reads."""

    def isatty(self) -> bool:
        return False

    def readline(self, *args: Any) -> str:  # type: ignore[override]
        raise AssertionError("stdin must not be read when it is not a terminal")


@pytest.fixture
def stdin(monkeypatch: pytest.MonkeyPatch):
    """Sets sys.stdin for the test: call it with a stream."""

    def set_stdin(stream: Any) -> Any:
        monkeypatch.setattr("sys.stdin", stream)
        return stream

    return set_stdin


def _write_template(tmp_path: Path, template: dict[str, Any]) -> str:
    path = tmp_path / "template.yaml"
    path.write_text(yaml.safe_dump(template), encoding="utf-8")
    return str(path)


def _python_step(name: str, **extra: Any) -> dict[str, Any]:
    return {
        "name": name,
        "script": {
            "actions": {"onRun": {"command": "python", "args": ["-c", f"print('{name} ran')"]}}
        },
        **extra,
    }


class TestRunPrompt:
    """The gate asks at the terminal when stdin is one."""

    def test_approve_at_prompt(self, capsys, stdin) -> None:
        # GIVEN
        terminal = stdin(FakeTerminal("y\n"))

        # WHEN
        outerr = run_openjd_cli_main(capsys, args=["run", APPROVALS_JOB], expected_exit_code=0)

        # THEN
        out = outerr.out
        # The gate opens once its dependency has completed, and the Steps that depend
        # on it run only after it is approved.
        assert out.index("Preview ran") < out.index(PROMPT) < out.index("Final ran")
        assert "Publish ran" in out
        assert "Independent ran" in out
        # The resolved message, and each link's label and full target, were shown.
        assert "Check the preview frames of sh010\n" in out
        assert "    before the final render runs.\n" in out
        assert "    - Preview frames: " + str(TEMPLATES_DIR / "out" / "preview") in out
        assert "    - Shot tracker: https://tracker.example.com/shots/sh010" in out
        # A local run does not apply the template's expiry.
        assert "Expires after: 3600 seconds (not applied by openjd run)" in out
        assert "Step 'ReviewPreview' approved at the prompt" in out
        assert (
            "Step 'ReviewPreview' is approval-only and has nothing to run; it completed successfully"
            in out
        )
        assert "Session ended successfully" in out
        assert "Approvals:\n  - 'ReviewPreview': approved at the prompt\n" in out
        # Only the four Steps with a script ran a Task.
        assert "Chunks run: 4" in out
        assert terminal.read() == ""

    @pytest.mark.parametrize("answer", ["n\n", "no\n", " NO \n"])
    def test_reject_at_prompt_blocks_dependents(self, capsys, stdin, answer: str) -> None:
        # GIVEN
        stdin(FakeTerminal(answer))

        # WHEN
        outerr = run_openjd_cli_main(capsys, args=["run", APPROVALS_JOB], expected_exit_code=1)

        # THEN
        out = outerr.out
        assert "Preview ran" in out
        # The rejected Step's dependents, direct and transitive, do not run...
        assert "Final ran" not in out
        assert "Publish ran" not in out
        assert (
            "Not running step 'Final': it depends on 'ReviewPreview', which did not complete successfully"
            in out
        )
        assert (
            "Not running step 'Publish': it depends on 'Final', which did not complete successfully"
            in out
        )
        # ...and the Steps that do not depend on it still do, as in a render farm.
        assert "Independent ran" in out
        assert (
            "Step 'ReviewPreview' rejected at the prompt. It and the Steps that depend on it will not run"
            in out
        )
        assert (
            "Not all Steps were approved. Steps that did not run: 'ReviewPreview', 'Final', 'Publish'"
            in out
        )
        assert (
            "Step 'ReviewPreview' was rejected at the prompt, so it did not run.\n"
            "Steps that did not run because they depend on a Step that was not approved: 'Final', 'Publish'.\n"
            in out
        )
        assert "Approvals:\n  - 'ReviewPreview': rejected at the prompt\n" in out
        assert "Chunks run: 2" in out

    def test_prompt_asks_again_until_answered(self, capsys, stdin) -> None:
        # GIVEN: a stray Enter key and an unknown answer are neither approval nor
        # rejection.
        stdin(FakeTerminal("\nmaybe\nyes\n"))

        # WHEN
        outerr = run_openjd_cli_main(capsys, args=["run", APPROVALS_JOB], expected_exit_code=0)

        # THEN
        assert outerr.out.count(PROMPT) == 3
        assert outerr.out.count("Please answer 'y' or 'n'.\n") == 2
        assert "Final ran" in outerr.out

    def test_end_of_input_at_prompt_rejects(self, capsys, stdin) -> None:
        # GIVEN
        stdin(FakeTerminal(""))

        # WHEN
        outerr = run_openjd_cli_main(capsys, args=["run", APPROVALS_JOB], expected_exit_code=1)

        # THEN
        assert outerr.out.count(PROMPT) == 1
        assert "Step 'ReviewPreview' rejected at the prompt" in outerr.out
        assert "Final ran" not in outerr.out

    def test_interrupted_prompt_stops_the_run(self, capsys, stdin) -> None:
        # GIVEN: Ctrl-C at the prompt. LocalSession's own SIGINT handler is installed
        # while the session runs, and it must interrupt the waiting prompt.
        class InterruptedTerminal(FakeTerminal):
            def readline(self, *args: Any) -> str:  # type: ignore[override]
                signal.raise_signal(signal.SIGINT)
                raise AssertionError("the SIGINT handler should have interrupted the prompt")

        stdin(InterruptedTerminal())

        # WHEN
        outerr = run_openjd_cli_main(capsys, args=["run", APPROVALS_JOB], expected_exit_code=1)

        # THEN
        out = outerr.out
        # The prompt's line is ended before the interruption is logged.
        assert PROMPT + "\n" in out
        assert "Interruption signal recieved." in out
        assert (
            "Session terminated by user while running action: 'Approve Step 'ReviewPreview''" in out
        )
        assert "The approval prompt for Step 'ReviewPreview' was interrupted" in out
        assert "ERROR executing action: 'Approve Step 'ReviewPreview''" in out
        # Ctrl-C stops the whole run, including the Steps that do not depend on the gate.
        assert "Final ran" not in out
        assert "Independent ran" not in out
        assert "Session ended with errors" in out

    def test_keyboard_interrupt_at_prompt_stops_the_run(self, capsys, stdin) -> None:
        # GIVEN: an interrupt that does not come through LocalSession's handler.
        class InterruptedTerminal(FakeTerminal):
            def readline(self, *args: Any) -> str:  # type: ignore[override]
                raise KeyboardInterrupt

        stdin(InterruptedTerminal())

        # WHEN
        outerr = run_openjd_cli_main(capsys, args=["run", APPROVALS_JOB], expected_exit_code=1)

        # THEN
        assert "The approval prompt for Step 'ReviewPreview' was interrupted" in outerr.out
        assert "Final ran" not in outerr.out
        assert "Session ended with errors" in outerr.out

    def test_prompt_goes_to_stderr_with_machine_readable_output(self, capsys, stdin) -> None:
        # GIVEN
        stdin(FakeTerminal("y\n"))

        # WHEN
        outerr = run_openjd_cli_main(
            capsys, args=["run", APPROVALS_JOB, "--output", "json"], expected_exit_code=0
        )

        # THEN: stdout is only the JSON result...
        result = json.loads(outerr.out)
        assert result["status"] == "success"
        assert result["approvals"] == [{"step_name": "ReviewPreview", "decision": "APPROVED"}]
        assert result["chunks_run"] == 4
        # ...and the request is shown along with the prompt on stderr, since the
        # Session log is not printed.
        assert "Step 'ReviewPreview' is approval-only." in outerr.err
        assert "    Check the preview frames of sh010" in outerr.err
        assert "    - Shot tracker: https://tracker.example.com/shots/sh010" in outerr.err
        assert PROMPT in outerr.err
        # The Session log still records the request and the decision.
        # (An empty log message is omitted from its entry.)
        logged = [entry.get("message") for entry in result["logs"]]
        assert "    Check the preview frames of sh010" in logged
        assert "Open Job Description CLI: Step 'ReviewPreview' approved at the prompt" in logged

    def test_template_text_is_shown_terminal_safe(self, capsys, stdin, tmp_path: Path) -> None:
        # GIVEN: a carriage return and a right-to-left override in a Job Parameter
        # value could make the prompt read differently from what it says.
        template = {
            "specificationVersion": "jobtemplate-2023-09",
            "extensions": ["STEP_APPROVALS"],
            "name": "Job",
            "parameterDefinitions": [{"name": "Shot", "type": "STRING"}],
            "steps": [
                {"name": "Gate", "approval": {"message": "Render shot {{Param.Shot}}?"}},
            ],
        }
        stdin(FakeTerminal("y\n"))

        # WHEN
        outerr = run_openjd_cli_main(
            capsys,
            args=["run", _write_template(tmp_path, template), "-p", "Shot=sh010\rOK" + chr(0x202E)],
            expected_exit_code=0,
        )

        # THEN
        assert "    Render shot sh010\\rOK\\u202e?\n" in outerr.out
        assert "\r" not in outerr.out
        assert chr(0x202E) not in outerr.out


class TestRunApproveStep:
    """--approve-step approves gates in advance."""

    def test_approve_step_skips_the_prompt(self, capsys, stdin) -> None:
        # GIVEN: a terminal that is never read.
        stdin(NotATerminal())

        # WHEN
        outerr = run_openjd_cli_main(
            capsys,
            args=["run", APPROVALS_JOB, "--approve-step", "ReviewPreview"],
            expected_exit_code=0,
        )

        # THEN
        out = outerr.out
        assert "Approve Step" not in out
        # The request is still recorded.
        assert "    Check the preview frames of sh010" in out
        assert "Step 'ReviewPreview' approved in advance (--approve-step)" in out
        assert "Final ran" in out
        assert "Publish ran" in out
        assert "Approvals:\n  - 'ReviewPreview': approved in advance (--approve-step)\n" in out

    def test_approve_step_with_interactive_terminal(self, capsys, stdin) -> None:
        # GIVEN
        terminal = stdin(FakeTerminal("n\n"))

        # WHEN
        outerr = run_openjd_cli_main(
            capsys,
            args=["run", APPROVALS_JOB, "--approve-step", "ReviewPreview"],
            expected_exit_code=0,
        )

        # THEN: the terminal is not asked.
        assert PROMPT not in outerr.out
        assert terminal.read() == "n\n"
        assert "Final ran" in outerr.out

    def test_approve_step_is_repeatable(self, capsys, stdin, tmp_path: Path) -> None:
        # GIVEN
        template = {
            "specificationVersion": "jobtemplate-2023-09",
            "extensions": ["STEP_APPROVALS"],
            "name": "Job",
            "steps": [
                {"name": "GateA", "approval": {"message": "A?"}},
                {"name": "GateB", "approval": {"message": "B?"}},
                _python_step("Work", dependencies=[{"dependsOn": "GateA"}, {"dependsOn": "GateB"}]),
            ],
        }
        stdin(NotATerminal())

        # WHEN
        outerr = run_openjd_cli_main(
            capsys,
            args=[
                "run",
                _write_template(tmp_path, template),
                "--approve-step",
                "GateA",
                "--approve-step",
                "GateB",
            ],
            expected_exit_code=0,
        )

        # THEN
        assert "Step 'GateA' approved in advance (--approve-step)" in outerr.out
        assert "Step 'GateB' approved in advance (--approve-step)" in outerr.out
        assert "Work ran" in outerr.out

    @pytest.mark.parametrize(
        "names,errors",
        [
            pytest.param(
                ["Nope"],
                ["No Step with name 'Nope' is defined in the given Job Template."],
                id="unknown step",
            ),
            pytest.param(
                ["Final"], ["Step 'Final' does not have an approval."], id="step without approval"
            ),
            pytest.param(
                ["ReviewPreview", "Nope", "Final"],
                [
                    "No Step with name 'Nope' is defined in the given Job Template.",
                    "Step 'Final' does not have an approval.",
                ],
                id="several",
            ),
        ],
    )
    def test_approve_step_errors(self, capsys, stdin, names: list[str], errors: list[str]) -> None:
        # GIVEN
        stdin(NotATerminal())
        args = ["run", APPROVALS_JOB]
        for name in names:
            args += ["--approve-step", name]

        # WHEN
        outerr = run_openjd_cli_main(capsys, args=args, expected_exit_code=1)

        # THEN: nothing runs.
        assert outerr.out.startswith(
            "Errors in --approve-step:" + "".join(f"\n - {error}" for error in errors) + "\n"
        )
        assert "Preview ran" not in outerr.out
        # The template's help, which lists the Steps that need approval, follows.
        assert "Steps That Need Approval (--approve-step STEP_NAME):\n  ReviewPreview" in outerr.out


class TestRunWithoutTerminal:
    """With no interactive terminal, a gate not approved in advance counts as rejected."""

    def test_no_terminal_rejects(self, capsys, stdin) -> None:
        # GIVEN
        stdin(NotATerminal())

        # WHEN
        outerr = run_openjd_cli_main(capsys, args=["run", APPROVALS_JOB], expected_exit_code=1)

        # THEN
        out = outerr.out
        assert "Approve Step" not in out
        assert "    Check the preview frames of sh010" in out
        assert (
            "Step 'ReviewPreview' not approved (no interactive terminal, and no --approve-step)."
            " It and the Steps that depend on it will not run" in out
        )
        assert (
            "Step 'ReviewPreview' was not approved: it needs approval, and there is no interactive"
            " terminal to ask at. Use --approve-step ReviewPreview to approve it in advance.\n"
            "Steps that did not run because they depend on a Step that was not approved: 'Final', 'Publish'.\n"
            in out
        )
        assert "Preview ran" in out
        assert "Independent ran" in out
        assert "Final ran" not in out
        assert "Publish ran" not in out

    @pytest.mark.parametrize("stream", [None, io.StringIO()], ids=["no stdin", "closed stdin"])
    def test_missing_or_closed_stdin_is_not_a_terminal(self, capsys, stdin, stream) -> None:
        # GIVEN
        if stream is not None:
            stream.close()
        stdin(stream)

        # WHEN
        outerr = run_openjd_cli_main(capsys, args=["run", APPROVALS_JOB], expected_exit_code=1)

        # THEN
        assert "not approved (no interactive terminal, and no --approve-step)" in outerr.out

    def test_hint_quotes_the_step_name(self, capsys, stdin, tmp_path: Path) -> None:
        # GIVEN
        template = {
            "specificationVersion": "jobtemplate-2023-09",
            "extensions": ["STEP_APPROVALS"],
            "name": "Job",
            "steps": [{"name": "Review the preview", "approval": {"message": "OK?"}}],
        }
        stdin(NotATerminal())

        # WHEN
        outerr = run_openjd_cli_main(
            capsys, args=["run", _write_template(tmp_path, template)], expected_exit_code=1
        )

        # THEN
        assert "Use --approve-step 'Review the preview' to approve it in advance." in outerr.out

    def test_json_result_of_a_rejection(self, capsys, stdin) -> None:
        # GIVEN
        stdin(NotATerminal())

        # WHEN
        outerr = run_openjd_cli_main(
            capsys, args=["run", APPROVALS_JOB, "--output", "json"], expected_exit_code=1
        )

        # THEN
        result = json.loads(outerr.out)
        assert result["status"] == "error"
        assert result["approvals"] == [{"step_name": "ReviewPreview", "decision": "NO_TERMINAL"}]
        assert result["steps_blocked"] == ["Final", "Publish"]
        assert result["chunks_run"] == 2
        assert outerr.err == ""


class TestRunApprovalWithScript:
    """An approval on a Step with a script gates its Tasks."""

    def test_approved_step_runs_its_tasks(self, capsys, stdin) -> None:
        # GIVEN
        stdin(FakeTerminal("y\n"))

        # WHEN
        outerr = run_openjd_cli_main(
            capsys, args=["run", APPROVAL_WITH_SCRIPT_JOB], expected_exit_code=0
        )

        # THEN
        out = outerr.out
        assert "Step 'UseLicensedAsset' needs approval before its Tasks run." in out
        # Resolved at job creation, with Step.Name and Job.Name (EXPR).
        assert (
            "    Confirm that the license for tree.usd covers UseLicensedAsset in LicensedJob."
            in out
        )
        assert out.index("approved at the prompt") < out.index("Processing frame 1")
        assert "Processing frame 2" in out
        assert "Chunks run: 2" in out

    def test_rejected_step_runs_no_tasks(self, capsys, stdin) -> None:
        # GIVEN
        stdin(FakeTerminal("n\n"))

        # WHEN
        outerr = run_openjd_cli_main(
            capsys, args=["run", APPROVAL_WITH_SCRIPT_JOB], expected_exit_code=1
        )

        # THEN
        assert "Processing frame" not in outerr.out
        assert "Running step 'UseLicensedAsset'" not in outerr.out
        assert (
            "Step 'UseLicensedAsset' was rejected at the prompt, so it did not run." in outerr.out
        )
        assert "Steps that did not run because" not in outerr.out
        assert "Step: UseLicensedAsset\n" in outerr.out
        assert "Chunks run: 0" in outerr.out

    def test_rejected_step_enters_no_step_environment(self, capsys, stdin, tmp_path: Path) -> None:
        # GIVEN
        template = {
            "specificationVersion": "jobtemplate-2023-09",
            "extensions": ["STEP_APPROVALS"],
            "name": "Job",
            "steps": [
                _python_step(
                    "Gated",
                    approval={"message": "Go?"},
                    stepEnvironments=[
                        {
                            "name": "StepEnv",
                            "script": {
                                "actions": {
                                    "onEnter": {
                                        "command": "python",
                                        "args": ["-c", "print('StepEnv entered')"],
                                    }
                                }
                            },
                        }
                    ],
                )
            ],
        }
        path = _write_template(tmp_path, template)
        stdin(NotATerminal())

        # WHEN
        rejected = run_openjd_cli_main(capsys, args=["run", path], expected_exit_code=1)
        approved = run_openjd_cli_main(
            capsys, args=["run", path, "--approve-step", "Gated"], expected_exit_code=0
        )

        # THEN
        assert "StepEnv entered" not in rejected.out
        assert "Gated ran" not in rejected.out
        assert approved.out.index("approved in advance") < approved.out.index("StepEnv entered")
        assert "Gated ran" in approved.out


class TestRunSelectedStep:
    """--step with approvals."""

    def test_selected_approval_only_step(self, capsys, stdin) -> None:
        # GIVEN
        stdin(FakeTerminal("y\n"))

        # WHEN
        outerr = run_openjd_cli_main(
            capsys, args=["run", APPROVALS_JOB, "--step", "ReviewPreview"], expected_exit_code=0
        )

        # THEN: approved, it counts as succeeded and runs nothing.
        assert "Step 'ReviewPreview' approved at the prompt" in outerr.out
        assert "ran\n" not in outerr.out
        assert "Session ended successfully" in outerr.out
        assert "Step: ReviewPreview\n" in outerr.out
        assert "Chunks run: 0" in outerr.out

    def test_selected_step_whose_dependency_is_rejected(self, capsys, stdin) -> None:
        # GIVEN
        stdin(FakeTerminal("n\n"))

        # WHEN
        outerr = run_openjd_cli_main(
            capsys,
            args=["run", APPROVALS_JOB, "--step", "Final", "--run-dependencies"],
            expected_exit_code=1,
        )

        # THEN
        assert "Preview ran" in outerr.out
        assert "Final ran" not in outerr.out
        assert (
            "Steps that did not run because they depend on a Step that was not approved: 'Final'."
            in outerr.out
        )

    def test_selected_step_without_its_dependencies_is_not_gated_by_them(
        self, capsys, stdin
    ) -> None:
        # GIVEN: the dependencies are not part of the run, as without approvals.
        stdin(NotATerminal())

        # WHEN
        outerr = run_openjd_cli_main(
            capsys, args=["run", APPROVALS_JOB, "--step", "Final"], expected_exit_code=0
        )

        # THEN
        assert "Final ran" in outerr.out
        assert "needs approval" not in outerr.out


class TestSummary:
    def test_job_summary(self, capsys) -> None:
        # WHEN
        outerr = run_openjd_cli_main(
            capsys, args=["summary", APPROVALS_JOB, "-p", "ShotName=sh020"], expected_exit_code=0
        )

        # THEN
        out = outerr.out
        # The approval-only Step has no Task.
        assert "Total steps: 5\nTotal tasks: 4\n" in out
        preview = str(TEMPLATES_DIR / "out" / "preview")
        assert (
            "2. 'ReviewPreview' (approval-only, no Tasks)\n"
            "  Needs approval:\n"
            "    Message:\n"
            "      Check the preview frames of sh020\n"
            "      before the final render runs.\n"
            "    Links:\n"
            f"      - Preview frames: {preview}\n"
            "      - Shot tracker: https://tracker.example.com/shots/sh020\n"
            "    Expires after: 3600 seconds\n"
            "    On reject: FAIL\n"
            "  1 dependencies\n"
        ) in out
        assert "3. 'Final' (1 total Tasks)\n  1 dependencies\n" in out

    def test_job_summary_of_approval_with_script(self, capsys) -> None:
        # WHEN
        outerr = run_openjd_cli_main(
            capsys, args=["summary", APPROVAL_WITH_SCRIPT_JOB], expected_exit_code=0
        )

        # THEN
        assert (
            "1. 'UseLicensedAsset' (2 total Tasks)\n"
            "  Needs approval:\n"
            "    Message:\n"
            "      Confirm that the license for tree.usd covers UseLicensedAsset in LicensedJob.\n"
            "    On reject: FAIL\n"
            "  Task parameters:\n"
        ) in outerr.out

    def test_step_summary(self, capsys) -> None:
        # WHEN
        outerr = run_openjd_cli_main(
            capsys,
            args=["summary", APPROVALS_JOB, "--step", "ReviewPreview"],
            expected_exit_code=0,
        )

        # THEN
        assert (
            "Total tasks: 0\n"
            "Total task parameters: 0\n"
            "Total environments: 0\n"
            "\n"
            "Needs approval (approval-only Step: it runs nothing):\n"
            "  Message:\n"
            "    Check the preview frames of sh010\n"
        ) in outerr.out
        assert "  On reject: FAIL\n\nDependencies (1):\n- 'Preview'\n" in outerr.out

    def test_json_summary(self, capsys) -> None:
        # WHEN
        outerr = run_openjd_cli_main(
            capsys, args=["summary", APPROVALS_JOB, "--output", "json"], expected_exit_code=0
        )

        # THEN
        steps = {step["name"]: step for step in json.loads(outerr.out)["steps"]}
        assert steps["ReviewPreview"]["approval_only"] is True
        assert "total_tasks" not in steps["ReviewPreview"]  # 0 is omitted
        assert steps["ReviewPreview"]["approval"] == {
            "message": "Check the preview frames of sh010\nbefore the final render runs.\n",
            "links": [
                {"label": "Preview frames", "target": str(TEMPLATES_DIR / "out" / "preview")},
                {"label": "Shot tracker", "target": "https://tracker.example.com/shots/sh010"},
            ],
            "expires_after_seconds": 3600,
            "on_reject": "FAIL",
        }
        assert "approval" not in steps["Final"]
        assert "approval_only" not in steps["Final"]

    def test_yaml_step_summary(self, capsys) -> None:
        # WHEN
        outerr = run_openjd_cli_main(
            capsys,
            args=[
                "summary",
                APPROVAL_WITH_SCRIPT_JOB,
                "--step",
                "UseLicensedAsset",
                "-p",
                "AssetName=rock.usd",
                "--output",
                "yaml",
            ],
            expected_exit_code=0,
        )

        # THEN
        result = yaml.safe_load(outerr.out)
        assert result["approval"] == {
            "message": "Confirm that the license for rock.usd covers UseLicensedAsset in LicensedJob.",
            "on_reject": "FAIL",
        }
        assert "approval_only" not in result


class TestCheck:
    def test_check_lists_the_steps_that_need_approval(self, capsys) -> None:
        # WHEN
        outerr = run_openjd_cli_main(capsys, args=["check", APPROVALS_JOB], expected_exit_code=0)

        # THEN
        assert outerr.out == (
            f"Template at '{APPROVALS_JOB}' passes validation checks.\n"
            "Steps that need approval before they run:\n"
            "  - 'ReviewPreview' (approval-only)\n"
        )

    def test_check_approval_with_script(self, capsys) -> None:
        # WHEN
        outerr = run_openjd_cli_main(
            capsys, args=["check", APPROVAL_WITH_SCRIPT_JOB], expected_exit_code=0
        )

        # THEN
        assert outerr.out.endswith(
            "Steps that need approval before they run:\n  - 'UseLicensedAsset'\n"
        )

    def test_check_without_approvals_is_unchanged(self, capsys) -> None:
        # WHEN
        template = str(TEMPLATES_DIR / "basic.yaml")
        outerr = run_openjd_cli_main(capsys, args=["check", template], expected_exit_code=0)

        # THEN
        assert outerr.out == f"Template at '{template}' passes validation checks.\n"

    def test_check_rejects_the_extension_when_not_accepted(self, capsys) -> None:
        # WHEN
        outerr = run_openjd_cli_main(
            capsys, args=["check", APPROVALS_JOB, "--extensions", "EXPR"], expected_exit_code=1
        )

        # THEN
        assert "Unsupported extension names: STEP_APPROVALS" in outerr.out

    def test_check_rejects_an_invalid_approval(self, capsys, tmp_path: Path) -> None:
        # GIVEN
        template = {
            "specificationVersion": "jobtemplate-2023-09",
            "extensions": ["STEP_APPROVALS"],
            "name": "Job",
            "steps": [
                {
                    "name": "Gate",
                    "approval": {"message": "OK?", "links": [{"label": "x", "url": "ftp://x"}]},
                }
            ],
        }

        # WHEN
        outerr = run_openjd_cli_main(
            capsys, args=["check", _write_template(tmp_path, template)], expected_exit_code=1
        )

        # THEN
        assert "url must be an absolute URI whose scheme is http or https" in outerr.out


class TestRunHelp:
    def test_help_lists_the_steps_that_need_approval(self, capsys, tmp_path: Path) -> None:
        # GIVEN
        template = {
            "specificationVersion": "jobtemplate-2023-09",
            "extensions": ["STEP_APPROVALS"],
            "name": "Job",
            "steps": [
                {"name": "Gate", "approval": {"message": "OK?"}},
                _python_step("Work", approval={"message": "Run it?"}),
                _python_step("Other"),
            ],
        }

        # WHEN
        outerr = run_openjd_cli_main(
            capsys,
            args=["run", _write_template(tmp_path, template), "--help"],
            expected_exit_code=0,
        )

        # THEN
        assert (
            "Steps That Need Approval (--approve-step STEP_NAME):\n"
            "  Gate (approval-only)\n"
            "  Work\n"
            "\n"
            "Standard Options:\n"
        ) in outerr.out
        assert "--approve-step STEP_NAME" in outerr.out.split("Standard Options:")[1]

    def test_help_without_approvals_has_no_approval_section(self, capsys) -> None:
        # WHEN
        outerr = run_openjd_cli_main(
            capsys, args=["run", str(TEMPLATES_DIR / "basic.yaml"), "--help"], expected_exit_code=0
        )

        # THEN
        assert "Steps That Need Approval" not in outerr.out


class TestHelpers:
    @pytest.mark.parametrize(
        "text,expected",
        [
            pytest.param("plain text", "plain text", id="plain"),
            pytest.param("two\nlines\tand a tab", "two\nlines\tand a tab", id="newline and tab"),
            pytest.param("crlf\r\nline", "crlf\nline", id="crlf"),
            pytest.param("over\rwrite", "over\\rwrite", id="carriage return"),
            pytest.param("\x1b[2Jcleared", "\\x1b[2Jcleared", id="escape sequence"),
            pytest.param("bell\x07", "bell\\x07", id="bell"),
            pytest.param("abc" + chr(0x202E) + "fed", "abc\\u202efed", id="right-to-left override"),
            pytest.param("x" + chr(0x2066) + "y", "x\\u2066y", id="isolate"),
            pytest.param("café 日本", "café 日本", id="other unicode"),
        ],
    )
    def test_terminal_safe(self, text: str, expected: str) -> None:
        assert terminal_safe(text) == expected

    def test_helpers_accept_models_without_approvals(self) -> None:
        # An openjd-model release without STEP_APPROVALS has no `approval` field.
        step = object()
        assert step_approval(step) is None
        assert is_approval_only(step) is False

    def test_run_task_action_refuses_an_approval_only_step(self) -> None:
        # GIVEN
        template = decode_job_template(
            template={
                "specificationVersion": "jobtemplate-2023-09",
                "extensions": ["STEP_APPROVALS"],
                "name": "Job",
                "steps": [{"name": "Gate", "approval": {"message": "OK?"}}],
            },
            supported_extensions=["STEP_APPROVALS"],
        )
        step = create_job(job_template=template, job_parameter_values={}).steps[0]
        assert step.script is None
        session = Mock()

        # WHEN / THEN
        with pytest.raises(RuntimeError, match="Step 'Gate' has no script to run."):
            RunTaskAction(session=session, step=step, parameters={}).run()
        session.run_task.assert_not_called()

    def test_gate_decisions(self) -> None:
        assert [decision.approved for decision in ApprovalDecision] == [True, True, False, False]
        assert StepApprovalGate(approved_in_advance=["A"]).decide("A", []) == (
            ApprovalDecision.APPROVED_IN_ADVANCE
        )

    def test_end_prompt_line_without_a_prompt_writes_nothing(self, capsys) -> None:
        StepApprovalGate().end_prompt_line()
        assert capsys.readouterr().out == ""
